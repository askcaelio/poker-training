"""Decision-making agents — bots and the human-player adapter.

An Agent's only job is `decide(view) -> Action`. They have no access to other
players' hole cards, only to what `HandView` exposes (own cards, public state,
action history).

Bot archetypes are deliberately simple and explicit, not GTO solvers:
  - Nit:           ultra-tight, never bluffs
  - TAG:           solid range, selective bluffs (textbook "good player")
  - LAG:           wider range, frequent aggression
  - Maniac:        bets/raises with almost anything
  - Station:       calls everything, almost never folds, almost never raises

Their value as opponents is *legibility*: you can read their ranges and
exploit them. Real poker training is largely about adjusting to player types,
so simple-and-readable beats unexploitable for our purposes.

Postflop decisions use Monte Carlo equity at low iteration counts (~1000) for
speed. That's ~2-3% precision, plenty for bot decisions.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .board_texture import analyze_texture
from .cards import Card, Rank
from .equity import equity_vs_random, equity_vs_range, equity_vs_ranges
from .game import Action, ActionType, HandView, Position, Street
from .opponent_model import OpponentTable, estimate_fold_equity
from .range_tracker import RangeTracker


# ─── Hand class strings ───────────────────────────────────────────────────────

def hand_class(c1: Card, c2: Card) -> str:
    """Convert two hole cards to a 169-class string: 'AKs', 'AKo', 'QQ'."""
    if c1.rank == c2.rank:
        return f"{c1.rank.char}{c2.rank.char}"
    high, low = (c1, c2) if c1.rank > c2.rank else (c2, c1)
    suited = c1.suit == c2.suit
    return f"{high.rank.char}{low.rank.char}{'s' if suited else 'o'}"


# ─── Preflop ranges (by archetype) ────────────────────────────────────────────

# Hands ordered roughly by strength (used for percent-based slicing).
# Top of the chart at index 0.
def _all_pairs() -> list[str]:
    return ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22"]


def _suited(min_high: str, min_low: str = "2") -> list[str]:
    """All suited hands with high card >= min_high (e.g. 'A' for any suited Ax)."""
    rank_chars = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    out = []
    for i, h in enumerate(rank_chars):
        for j in range(i):
            l = rank_chars[j]
            if rank_chars.index(h) >= rank_chars.index(min_high) and \
               rank_chars.index(l) >= rank_chars.index(min_low):
                out.append(f"{h}{l}s")
    return out


# Range definitions live in preflop_ranges.py to break circular imports.
# Re-exported here for backward compatibility.
from .preflop_ranges import (
    LAG_RANGE, MANIAC_RANGE, NIT_RANGE, STATION_RANGE, TAG_RANGE,
)


# ─── Agent base class ─────────────────────────────────────────────────────────

class Agent(ABC):
    """Abstract decision-maker. Subclasses implement `decide(view)`."""

    name: str

    @abstractmethod
    def decide(self, view: HandView) -> Action:
        """Return the action this agent wants to take given the view."""
        ...


# ─── Bot agents ───────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    """Tunables that drive a bot's decisions.

    Defaults are TAG. Other archetypes override.
    """
    name: str = "TAG"
    preflop_range: set[str] = None  # type: ignore
    open_size_bb: float = 2.5       # how many BBs to raise when opening
    threebet_size_mult: float = 3.0 # 3-bet to N× the open
    cbet_pct: float = 0.66          # cbet size as fraction of pot
    value_threshold: float = 0.55   # equity above which to bet for value
    bluff_freq: float = 0.20        # P(bluff with weak hand on dry boards)
    fold_threshold_buffer: float = 0.0  # extra equity needed to call beyond pot odds
    equity_iters: int = 800         # MC iterations per decision (lower = faster, ~3% precision)
    aggression: float = 0.5         # P(raise vs call) when both look fine
    prefer_limp: bool = False       # if True, opens with call rather than raise (Station-style)
    uses_opponent_model: bool = True   # smart archetypes adapt to villain stats; Maniac/Station ignore

    def __post_init__(self):
        if self.preflop_range is None:
            self.preflop_range = TAG_RANGE


class BotAgent(Agent):
    """Generic bot driven by a BotConfig. Use the archetype constructors below."""

    def __init__(
        self,
        name: str,
        config: BotConfig,
        rng: Optional[random.Random] = None,
        opponent_table: Optional[OpponentTable] = None,
        range_tracker: Optional[RangeTracker] = None,
    ):
        self.name = name
        self.config = config
        self.rng = rng if rng is not None else random.Random()
        # Opponent stats for fold-equity-aware decisions. Optional — when None,
        # bot falls back to the simpler bluff_freq / value_threshold logic.
        self.opponent_table = opponent_table
        # Per-hand range tracker. The simulator resets this each hand and
        # narrows ranges on each action. When present, postflop equity is
        # computed against the inferred range, not random.
        self.range_tracker = range_tracker

    def decide(self, view: HandView) -> Action:
        if view.street == Street.PREFLOP:
            return self._decide_preflop(view)
        return self._decide_postflop(view)

    # ─── Preflop ───────────────────────────────────────────────────────────

    def _decide_preflop(self, view: HandView) -> Action:
        hand = hand_class(*view.your_hole)

        # Position-aware range: UTG/MP play tighter than BTN.
        # When facing a raise, BB and SB use the full archetype range to defend
        # (they have to play wider to stop being exploited).
        from .archetype_ranges import position_aware_open_set
        pos = view.your_position.value if view.your_position else "BTN"
        facing_raise = view.current_bet > 1.0
        if facing_raise and pos in ("BB", "SB"):
            applicable_range = self.config.preflop_range
        else:
            applicable_range = position_aware_open_set(self.config.preflop_range, pos)
        in_range = hand in applicable_range

        if not in_range:
            if view.to_call == 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.FOLD)

        # In our range. If no one's raised yet → open. If facing a raise → call/3bet.
        if not facing_raise:
            if view.to_call == 0:
                # Check option (BB unraised) — small chance to raise
                if self.rng.random() < self.config.aggression * 0.5:
                    return self._raise_to(view, view.pot * 3)
                return Action(ActionType.CHECK)
            # Stations limp (call) with wide range; aggressive players open raise.
            is_premium = hand in {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
            if self.config.prefer_limp and not is_premium:
                return self._call_or_check(view)
            target = self.config.open_size_bb  # in chips, assuming BB=1
            return self._raise_to(view, target)

        # Facing a raise: call by default; sometimes 3-bet with premium
        is_premium = hand in {"AA", "KK", "QQ", "AKs", "AKo"}
        if is_premium and self.rng.random() < 0.7:
            target = view.current_bet * self.config.threebet_size_mult
            return self._raise_to(view, target)
        return self._call_or_check(view)

    # ─── Postflop ──────────────────────────────────────────────────────────

    def _decide_postflop(self, view: HandView) -> Action:
        n_opps = max(1, view.num_active_opponents)

        # Speed shortcut: if hero has a monster (set or better), no need to
        # compute equity — just play it for value. Saves the MC entirely on
        # the strongest hands.
        from .evaluator import evaluate as _eval
        try:
            hand_strength = _eval(list(view.your_hole), list(view.board))
            # category_index <= 6 means Three of a Kind or stronger (treys convention)
            is_monster = hand_strength.category_index <= 6
        except ValueError:
            is_monster = False

        if is_monster:
            eq = 0.95   # don't bother with MC — it's a value bet either way
        else:
            eq = self._compute_postflop_equity(view, n_opps)

        # River gets specialized polarized betting logic — value/bluff/check
        # rather than a single equity threshold.
        if view.street == Street.RIVER:
            return self._decide_river(view, eq)

        if view.to_call == 0:
            return self._decide_unraised(view, eq)
        return self._decide_facing_bet(view, eq)

    def _decide_unraised(self, view: HandView, eq: float) -> Action:
        """Open or check. Uses fold equity if opponent_table available."""
        bet_size = view.pot * self._sizing_for_board(view)

        if self.opponent_table is None or not self.config.uses_opponent_model:
            # Legacy path: equity-only logic. Used by Maniac/Station who don't adapt.
            if eq >= self.config.value_threshold:
                return self._bet(view, bet_size)
            if self.rng.random() < self.config.bluff_freq:
                return self._bet(view, bet_size)
            return Action(ActionType.CHECK)

        # Strong hands always value-bet — the equity-vs-random underestimates
        # equity vs villain's calling range, so the EV math undersells value bets.
        # Fold-equity logic only applies to marginal/bluff decisions.
        if eq >= self.config.value_threshold:
            return self._bet(view, bet_size)

        # Marginal/weak hand: use fold-equity math to decide bluff vs check.
        fold_eq = self._avg_fold_equity(view, context="cbet")

        # EV(check) ≈ eq * pot   (rough: assume showdown, no further betting)
        ev_check = eq * view.pot
        # EV(bet) = fold_eq * pot + (1 - fold_eq) * (eq * (pot + 2*bet) - bet)
        ev_bet = (
            fold_eq * view.pot
            + (1 - fold_eq) * (eq * (view.pot + 2 * bet_size) - bet_size)
        )

        if ev_bet > ev_check + 0.5:  # small bias toward checking on ties
            return self._bet(view, bet_size)
        return Action(ActionType.CHECK)

    def _decide_facing_bet(self, view: HandView, eq: float) -> Action:
        """Fold, call, or raise. Uses fold equity, SPR, and implied odds."""
        required_eq = view.to_call / (view.pot + view.to_call)
        # Implied odds: drawing hands deserve a discount when stacks are deep.
        effective_required_eq = self._apply_implied_odds(view, required_eq, eq)
        equity_edge = eq - effective_required_eq
        raise_total = view.current_bet + view.pot * 1.0

        spr = self._spr(view)

        # SPR-aware adjustments:
        # Low SPR (committed) → wider raises and calls; one-pair is enough to commit
        # High SPR (deep) → tighter raises (don't get one-pair stacks in)
        if spr < 2.0:
            # Committed — get it in with anything decent
            if eq > effective_required_eq:
                return self._raise_to(view, raise_total)
        elif spr > 6.0 and view.street != Street.RIVER:
            # Deep — only raise with strong hands
            if equity_edge > 0.25 and self.rng.random() < self.config.aggression:
                return self._raise_to(view, raise_total)
        else:
            # Standard SPR
            if equity_edge > 0.15 and self.rng.random() < self.config.aggression:
                return self._raise_to(view, raise_total)

        # Marginal but priced-in (or implied-priced-in) → call
        if equity_edge > self.config.fold_threshold_buffer:
            return self._call_or_check(view)

        # Weak hand below pot odds → fold or bluff-raise.
        if self.opponent_table is not None and self.config.uses_opponent_model:
            bettor_id = self._last_aggressor_id(view)
            if bettor_id is not None:
                fold_eq = estimate_fold_equity(
                    bettor_id, self.opponent_table, context="barrel",
                )
                bet_cost = raise_total - view.your_bet_this_round
                required_fold_eq = bet_cost / (bet_cost + view.pot + view.to_call)
                if fold_eq > required_fold_eq + 0.05:
                    return self._raise_to(view, raise_total)
            return Action(ActionType.FOLD)

        if self.rng.random() < self.config.bluff_freq:
            return self._raise_to(view, raise_total)
        return Action(ActionType.FOLD)

    # ─── River: polarized value vs bluff ────────────────────────────────────

    def _decide_river(self, view: HandView, eq: float) -> Action:
        """River-specific polarized logic: value bet very strong, bluff very
        weak (with fold equity, preferring blocker hands), check the middle.
        """
        if view.to_call > 0:
            return self._decide_facing_bet(view, eq)

        bet_size = view.pot * self._sizing_for_board(view) * 1.1

        # Value zone: very strong hands → bet for value
        if eq >= 0.75:
            return self._bet(view, bet_size)

        # Bluff zone: very weak hands with fold equity → bluff
        # Prefer hands that BLOCK villain's value range (e.g., nut flush blocker
        # on a flush board makes the bluff much more credible because villain
        # can't have the nuts as often).
        if eq < 0.20 and self.opponent_table is not None and self.config.uses_opponent_model:
            fold_eq = self._avg_fold_equity(view, context="barrel")
            required_fold_eq = bet_size / (bet_size + view.pot)
            has_blocker = self._has_blocker_to_villain_value(view)
            # Blocker hands need less fold equity buffer; non-blockers need more
            buffer = 0.0 if has_blocker else 0.10
            if fold_eq > required_fold_eq + buffer:
                return self._bet(view, bet_size)

        # Middle: showdown value — check it down
        return Action(ActionType.CHECK)

    def _has_blocker_to_villain_value(self, view: HandView) -> bool:
        """True if hero holds a card that blocks villain's nut value hand.

        Currently detects: on a 3+ flush-suit board, hero holds the A of
        that suit → villain can't have the nut flush. Strongest blocker effect.
        """
        if not view.board:
            return False
        # Find dominant suit if board is 3+ flush
        suit_counts: dict = {}
        for c in view.board:
            suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
        if not suit_counts:
            return False
        dominant_suit = max(suit_counts.keys(), key=lambda s: suit_counts[s])
        if suit_counts[dominant_suit] < 3:
            return False
        # Hero blocks if they hold the Ace of that suit (or a high card)
        for c in view.your_hole:
            if c.suit == dominant_suit and c.rank == Rank.ACE:
                return True
        return False

    # ─── SPR + implied odds ─────────────────────────────────────────────────

    def _spr(self, view: HandView) -> float:
        """Stack-to-Pot Ratio: effective stack ÷ pot.

        Effective stack = min(your remaining stack, deepest active villain's stack).
        Low SPR (< 2): committed. Standard (2-6): play normally. Deep (> 6): careful.
        """
        if view.pot <= 0:
            return 99.0
        active_villain_stacks = [o.stack for o in view.others if not o.folded]
        if not active_villain_stacks:
            return 99.0
        effective = min(view.your_stack, max(active_villain_stacks))
        return effective / view.pot

    def _apply_implied_odds(
        self, view: HandView, required_eq: float, eq: float,
    ) -> float:
        """Adjust required equity for both implied odds (draws in deep games
        get a discount) and reverse implied odds (vulnerable made hands on
        wet boards face a penalty).
        """
        if view.street == Street.RIVER or not view.board:
            return required_eq
        spr = self._spr(view)
        if spr < 2.0:
            return required_eq

        # Forward implied: drawing hands get a discount
        is_drawing = 0.25 <= eq <= 0.45
        if is_drawing:
            implied_factor = min(spr * 0.05, 0.15)
            return required_eq * (1 - implied_factor)

        # Reverse implied: vulnerable made hands (one pair) on wet boards
        # face a penalty — even when ahead now, future bets cost us more
        # than they gain when we get there safely.
        if eq > 0.55 and self._is_one_pair_on_wet_board(view):
            penalty_factor = min(spr * 0.04, 0.12)
            return required_eq * (1 + penalty_factor)

        return required_eq

    def _is_one_pair_on_wet_board(self, view: HandView) -> bool:
        """True if hero has exactly one pair on a wet board — vulnerable spot."""
        if not view.board:
            return False
        from .evaluator import evaluate as _eval
        from .board_texture import analyze_texture
        try:
            hs = _eval(list(view.your_hole), list(view.board))
            tex = analyze_texture(list(view.board))
        except ValueError:
            return False
        # Pair-only AND wet board (lots of draws to outdraw us)
        return hs.category_index == 8 and tex.is_wet

    def _compute_postflop_equity(self, view: HandView, n_opps: int) -> float:
        """Pick the best equity calc for the situation.

        Decision tree:
          1. If range tracker available AND all active opponents have tracked
             ranges → equity_vs_ranges (full multiway accuracy)
          2. Else if heads-up AND tracked range available → equity_vs_range
          3. Else → equity_vs_random (legacy fallback)
        """
        active = [o for o in view.others if not o.folded]

        if self.range_tracker is not None and self.config.uses_opponent_model and active:
            tracked_ranges = []
            all_have_ranges = True
            for o in active:
                r = self.range_tracker.get(o.id)
                if r is None or r.total_combos == 0:
                    all_have_ranges = False
                    break
                tracked_ranges.append(r)

            if all_have_ranges:
                try:
                    if len(tracked_ranges) == 1:
                        return equity_vs_range(
                            hole=list(view.your_hole),
                            villain_range=tracked_ranges[0],
                            board=list(view.board),
                            iterations=self.config.equity_iters,
                            rng=self.rng,
                        ).equity_pct / 100.0
                    return equity_vs_ranges(
                        hole=list(view.your_hole),
                        opponent_ranges=tracked_ranges,
                        board=list(view.board),
                        iterations=self.config.equity_iters,
                        rng=self.rng,
                    ).equity_pct / 100.0
                except ValueError:
                    pass

        # Fallback: equity vs random
        try:
            return equity_vs_random(
                hole=list(view.your_hole),
                board=list(view.board),
                num_opponents=n_opps,
                iterations=self.config.equity_iters,
                rng=self.rng,
            ).equity_pct / 100.0
        except ValueError:
            return 0.0

    def _avg_fold_equity(self, view: HandView, context: str) -> float:
        """Average fold-equity across active opponents (those still in the hand)."""
        if self.opponent_table is None:
            return 0.4
        active = [o for o in view.others if not o.folded]
        if not active:
            return 0.0
        fold_eqs = [
            estimate_fold_equity(o.id, self.opponent_table, context)
            for o in active
        ]
        # Use the *minimum* fold-eq, not average — only takes one stubborn caller
        # to make a bluff fail. This is closer to multiway reality than averaging.
        return min(fold_eqs)

    def _last_aggressor_id(self, view: HandView) -> Optional[str]:
        """Identify the player who made the most recent bet/raise this street."""
        for rec in reversed(view.action_history):
            if rec.street == view.street and rec.action.type in (ActionType.BET, ActionType.RAISE):
                return rec.player_id
        return None

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _call_or_check(self, view: HandView) -> Action:
        if view.to_call == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, amount=min(view.to_call, view.your_stack))

    def _bet(self, view: HandView, target_amount: float) -> Action:
        amount = max(target_amount, 1.0)  # at least BB
        amount = min(amount, view.your_stack)
        return Action(ActionType.BET, amount=amount)

    def _sizing_for_board(self, view: HandView) -> float:
        """Return bet size as fraction of pot, adapted to board texture.

        Base = self.config.cbet_pct. Wet boards get bigger sizing (charge draws);
        dry boards get smaller (probe + thin value). Paired boards stay near base.
        """
        base = self.config.cbet_pct
        if not view.board:
            return base
        try:
            tex = analyze_texture(list(view.board))
        except ValueError:
            return base
        if tex.is_wet:
            return min(base * 1.3, 1.0)   # 30% larger, capped at pot-sized
        if tex.is_dry:
            return max(base * 0.7, 0.33)  # 30% smaller, floored at 1/3 pot
        return base

    def _raise_to(self, view: HandView, raise_to_total: float) -> Action:
        """Raise such that bet_this_round becomes raise_to_total."""
        contribution = raise_to_total - view.your_bet_this_round
        contribution = min(contribution, view.your_stack)
        if contribution <= view.to_call:
            # Can't raise — degrade to call
            return self._call_or_check(view)
        return Action(ActionType.RAISE, amount=contribution)


# ─── Archetype constructors ───────────────────────────────────────────────────

def make_nit(name: str = "Nit", rng: Optional[random.Random] = None) -> BotAgent:
    return BotAgent(name, BotConfig(
        name="Nit", preflop_range=NIT_RANGE,
        cbet_pct=0.5, value_threshold=0.65, bluff_freq=0.0,
        fold_threshold_buffer=0.05, aggression=0.3,
    ), rng=rng)


def make_tag(name: str = "TAG", rng: Optional[random.Random] = None) -> BotAgent:
    return BotAgent(name, BotConfig(
        name="TAG", preflop_range=TAG_RANGE,
        value_threshold=0.55, bluff_freq=0.20,
        fold_threshold_buffer=0.0, aggression=0.6,
    ), rng=rng)


def make_lag(name: str = "LAG", rng: Optional[random.Random] = None) -> BotAgent:
    return BotAgent(name, BotConfig(
        name="LAG", preflop_range=LAG_RANGE, open_size_bb=3.0,
        cbet_pct=0.75, value_threshold=0.50, bluff_freq=0.35,
        fold_threshold_buffer=-0.03, aggression=0.75,
    ), rng=rng)


def make_maniac(name: str = "Maniac", rng: Optional[random.Random] = None) -> BotAgent:
    # Maniacs DON'T adapt — they bet wildly regardless of opponent. That's the trait.
    return BotAgent(name, BotConfig(
        name="Maniac", preflop_range=MANIAC_RANGE, open_size_bb=4.0,
        cbet_pct=1.0, value_threshold=0.45, bluff_freq=0.55,
        fold_threshold_buffer=-0.10, aggression=0.90,
        uses_opponent_model=False,
    ), rng=rng)


def make_station(name: str = "Station", rng: Optional[random.Random] = None) -> BotAgent:
    # Stations DON'T adapt — they call regardless of opponent profile. That's the trait.
    return BotAgent(name, BotConfig(
        name="Station", preflop_range=STATION_RANGE, open_size_bb=2.5,
        cbet_pct=0.5, value_threshold=0.65, bluff_freq=0.02,
        fold_threshold_buffer=-0.20,  # calls way below pot odds
        aggression=0.10, prefer_limp=True,
        uses_opponent_model=False,
    ), rng=rng)


# ─── Random agent (for testing / baseline) ────────────────────────────────────

class RandomAgent(Agent):
    """Picks any legal action uniformly. Useful as a sanity baseline."""

    def __init__(self, name: str = "Random", rng: Optional[random.Random] = None):
        self.name = name
        self.rng = rng if rng is not None else random.Random()

    def decide(self, view: HandView) -> Action:
        # Need to query legal actions — but we only have view, not state.
        # Use a heuristic: if to_call == 0, choose check or bet; else fold/call/raise.
        if view.to_call == 0:
            choice = self.rng.choice(["check", "bet"])
            if choice == "check":
                return Action(ActionType.CHECK)
            return Action(ActionType.BET, amount=min(view.pot * 0.5 + 1.0, view.your_stack))
        choice = self.rng.choice(["fold", "call", "raise"])
        if choice == "fold":
            return Action(ActionType.FOLD)
        if choice == "call":
            return Action(ActionType.CALL, amount=min(view.to_call, view.your_stack))
        # raise
        contribution = max(view.min_raise, view.to_call * 2)
        contribution = min(contribution, view.your_stack)
        if contribution <= view.to_call:
            return Action(ActionType.CALL, amount=min(view.to_call, view.your_stack))
        return Action(ActionType.RAISE, amount=contribution)


# ─── Always-fold agent (useful for some tests) ────────────────────────────────

class AlwaysFoldAgent(Agent):
    """Folds to any bet, checks otherwise."""
    name = "AlwaysFold"

    def decide(self, view: HandView) -> Action:
        if view.to_call == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)
