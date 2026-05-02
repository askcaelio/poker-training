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

from .cards import Card, Rank
from .equity import equity_vs_random
from .game import Action, ActionType, HandView, Position, Street
from .opponent_model import OpponentTable, estimate_fold_equity


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


# Explicit ranges. These are textbook approximations, not GTO.
NIT_RANGE: set[str] = {
    "AA", "KK", "QQ", "JJ", "TT", "99",
    "AKs", "AKo", "AQs", "AQo", "AJs",
    "KQs",
}  # ~6% of hands

TAG_RANGE: set[str] = NIT_RANGE | {
    "88", "77", "66", "55", "44", "33", "22",
    "AJo", "ATs", "ATo", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQo", "KJs", "KJo", "KTs", "K9s",
    "QJs", "QJo", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s", "97s", "87s", "86s", "76s", "75s", "65s", "54s",
}  # ~18-20%

LAG_RANGE: set[str] = TAG_RANGE | {
    "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "Q8s", "Q7s", "Q6s", "Q5s",
    "J8s", "J7s",
    "T7s", "T6s",
    "96s", "85s", "74s", "64s", "53s", "43s",
    "ATo", "K9o", "Q9o", "J9o", "T9o", "98o", "87o", "76o",
}  # ~30-32%

MANIAC_RANGE: set[str] = LAG_RANGE | {
    "Q4s", "Q3s", "Q2s", "J6s", "J5s", "J4s", "J3s", "J2s",
    "T5s", "T4s", "T3s", "T2s",
    "95s", "94s", "93s", "92s", "84s", "83s", "82s", "73s", "72s",
    "63s", "62s", "52s", "42s", "32s",
    "K8o", "K7o", "K6o", "Q8o", "Q7o", "J8o", "T8o", "97o", "86o", "75o", "65o", "54o",
}  # ~50%

# Station plays almost everything (just folds the worst trash if facing a big bet)
STATION_RANGE: set[str] = MANIAC_RANGE | {
    "K5o", "K4o", "K3o", "K2o", "Q6o", "Q5o", "Q4o", "Q3o", "Q2o",
    "J7o", "J6o", "J5o", "J4o", "J3o", "J2o",
    "T7o", "T6o", "T5o", "T4o", "T3o", "T2o",
    "96o", "95o", "94o", "93o", "92o",
    "85o", "84o", "83o", "82o", "74o", "73o", "72o",
    "64o", "63o", "62o", "53o", "52o", "42o", "32o",
}  # ~85%


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
    ):
        self.name = name
        self.config = config
        self.rng = rng if rng is not None else random.Random()
        # Opponent stats for fold-equity-aware decisions. Optional — when None,
        # bot falls back to the simpler bluff_freq / value_threshold logic.
        self.opponent_table = opponent_table

    def decide(self, view: HandView) -> Action:
        if view.street == Street.PREFLOP:
            return self._decide_preflop(view)
        return self._decide_postflop(view)

    # ─── Preflop ───────────────────────────────────────────────────────────

    def _decide_preflop(self, view: HandView) -> Action:
        hand = hand_class(*view.your_hole)
        in_range = hand in self.config.preflop_range

        # Already a raise in front?
        facing_raise = view.current_bet > 1.0  # heuristic: BB = 1.0

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
        # Compute equity vs random opponents (cheap MC)
        n_opps = max(1, view.num_active_opponents)
        try:
            eq = equity_vs_random(
                hole=list(view.your_hole),
                board=list(view.board),
                num_opponents=n_opps,
                iterations=self.config.equity_iters,
                rng=self.rng,
            ).equity_pct / 100.0
        except ValueError:
            eq = 0.0

        if view.to_call == 0:
            return self._decide_unraised(view, eq)
        return self._decide_facing_bet(view, eq)

    def _decide_unraised(self, view: HandView, eq: float) -> Action:
        """Open or check. Uses fold equity if opponent_table available."""
        bet_size = view.pot * self.config.cbet_pct

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
        """Fold, call, or raise. Uses fold equity for raise-as-bluff decisions."""
        required_eq = view.to_call / (view.pot + view.to_call)
        equity_edge = eq - required_eq
        raise_total = view.current_bet + view.pot * 1.0

        # Strong hand → raise for value (with fold-eq blending the EV when available)
        if equity_edge > 0.15 and self.rng.random() < self.config.aggression:
            return self._raise_to(view, raise_total)

        # Marginal but priced-in → call
        if equity_edge > self.config.fold_threshold_buffer:
            return self._call_or_check(view)

        # Weak hand below pot odds → fold or bluff-raise.
        # Bluff-raise EV requires significant fold equity vs the bettor.
        if self.opponent_table is not None and self.config.uses_opponent_model:
            # Find the bettor — heuristic: the most recent aggressor in this round
            bettor_id = self._last_aggressor_id(view)
            if bettor_id is not None:
                fold_eq = estimate_fold_equity(
                    bettor_id, self.opponent_table, context="barrel",
                )
                # Bluff-raise EV: roughly fold_eq * (pot + their_bet) - bet_cost
                # Required fold_eq for breakeven: bet_cost / (bet_cost + pot)
                bet_cost = raise_total - view.your_bet_this_round
                required_fold_eq = bet_cost / (bet_cost + view.pot + view.to_call)
                if fold_eq > required_fold_eq + 0.05:
                    return self._raise_to(view, raise_total)
            return Action(ActionType.FOLD)

        # No opponent model — fall back to random bluff frequency
        if self.rng.random() < self.config.bluff_freq:
            return self._raise_to(view, raise_total)
        return Action(ActionType.FOLD)

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
