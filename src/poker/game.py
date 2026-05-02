"""Game state primitives for No-Limit Texas Hold'em.

Pure state machine: no I/O, no players-as-agents, no UI. Just types and the
lifecycle functions that move a hand from preflop to showdown.

Player decision-making is *not* in this module. The agent abstraction
(`HumanPlayer`, `BotPlayer`, etc.) sits on top of this and supplies actions
via `apply_action()`. That separation lets the same engine power autonomous
bot-vs-bot simulations and interactive human play.

Conventions:
- Positions are 6-max NLHE: SB, BB, UTG, MP, CO, BTN. We support 2-9 seats
  by using a subset (e.g., 2-handed = SB/BB only).
- All amounts are floats (chips). Use whatever unit you like (BBs, dollars).
- "Pot" includes all chips committed across all completed streets PLUS the
  current street's bets. Side pots are computed at award time.
- Lower treys-rank = stronger hand (we keep that convention internally).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .cards import Card, Deck


# ─── Enums ────────────────────────────────────────────────────────────────────

class Position(Enum):
    """Seat position in 6-max NLHE. Order matters: action moves clockwise from SB
    preflop (UTG acts first), and from SB postflop."""
    SB = "SB"      # small blind
    BB = "BB"      # big blind
    UTG = "UTG"    # under the gun
    MP = "MP"      # middle position
    CO = "CO"      # cutoff
    BTN = "BTN"    # button


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"          # opens betting (no prior bet this street)
    RAISE = "raise"      # increases an existing bet


# ─── Action ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    """A betting action.

    For BET / RAISE / CALL: `amount` is the number of CHIPS the player puts in
    (not the raise-to total). Validation against legal-actions happens in
    apply_action().

    Examples:
        Action(ActionType.FOLD)
        Action(ActionType.CHECK)
        Action(ActionType.CALL, amount=20)        # call $20
        Action(ActionType.BET, amount=50)         # bet $50 into a checked-around pot
        Action(ActionType.RAISE, amount=80)       # raise — total chips going in
    """

    type: ActionType
    amount: float = 0.0

    def __str__(self) -> str:
        if self.type in (ActionType.FOLD, ActionType.CHECK):
            return self.type.value
        return f"{self.type.value} {self.amount:g}"


# ─── Player (game-state participant) ──────────────────────────────────────────

@dataclass
class Player:
    """A seat at the table for one hand. The decision-maker (HumanPlayer / BotPlayer)
    is a separate concept — this dataclass is just the chips, cards, and per-hand
    state that the engine tracks.
    """
    id: str
    stack: float
    position: Optional[Position] = None
    hole_cards: Optional[tuple[Card, Card]] = None
    folded: bool = False
    all_in: bool = False
    bet_this_round: float = 0.0   # chips committed this betting round
    total_invested: float = 0.0   # chips committed this hand (for side-pot math)

    @property
    def in_hand(self) -> bool:
        """True if still eligible to win the pot (not folded)."""
        return not self.folded

    @property
    def can_act(self) -> bool:
        """True if engine should ask this player for an action."""
        return self.in_hand and not self.all_in and self.stack > 0


# ─── Action records (history) ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionRecord:
    player_id: str
    action: Action
    street: Street
    pot_before: float
    pot_after: float


# ─── Player view (what a single player can see) ───────────────────────────────

@dataclass(frozen=True)
class OtherPlayerView:
    id: str
    position: Optional[Position]
    stack: float
    bet_this_round: float
    total_invested: float
    folded: bool
    all_in: bool


@dataclass(frozen=True)
class HandView:
    """Read-only slice of the hand state visible to one player.

    The decision-maker uses this to choose an action. It deliberately omits
    other players' hole cards.
    """
    your_id: str
    your_hole: tuple[Card, Card]
    your_stack: float
    your_position: Optional[Position]
    your_bet_this_round: float

    board: tuple[Card, ...]
    street: Street
    pot: float
    current_bet: float                # the bet level this round
    to_call: float                    # what you need to add to call
    min_raise: float                  # minimum legal raise (chips total going in)

    others: tuple[OtherPlayerView, ...]
    action_history: tuple[ActionRecord, ...]

    @property
    def num_active_opponents(self) -> int:
        return sum(1 for p in self.others if not p.folded)


# ─── Hand state ───────────────────────────────────────────────────────────────

@dataclass
class HandState:
    """Mutable state of one in-progress hand.

    Use the lifecycle functions (start_hand, advance_street, etc.) rather than
    poking fields directly. apply_action() is the canonical mutator.
    """
    players: list[Player]               # in seat order (clockwise around table)
    button_index: int                   # which seat has the button
    deck: Deck
    blinds: tuple[float, float]         # (small_blind, big_blind)
    board: list[Card] = field(default_factory=list)
    street: Street = Street.PREFLOP
    pot: float = 0.0
    current_bet: float = 0.0            # max bet_this_round across players
    last_raise_size: float = 0.0        # the size of the last raise (for min-raise)
    actor_index: int = 0
    actions: list[ActionRecord] = field(default_factory=list)
    # Players who still need to act this betting round. A bet/raise resets this
    # to "everyone except the aggressor". When empty, round completes.
    players_to_act: set[str] = field(default_factory=set)

    @property
    def num_in_hand(self) -> int:
        return sum(1 for p in self.players if p.in_hand)

    @property
    def is_complete(self) -> bool:
        """Hand ends when ≤1 player remains or we've finished river betting."""
        return self.num_in_hand <= 1 or self.street == Street.SHOWDOWN

    def actor(self) -> Player:
        return self.players[self.actor_index]

    def view_for(self, player_id: str) -> HandView:
        """Build the read-only view a given player would see right now."""
        you: Optional[Player] = None
        others: list[OtherPlayerView] = []
        for p in self.players:
            if p.id == player_id:
                you = p
            else:
                others.append(OtherPlayerView(
                    id=p.id, position=p.position, stack=p.stack,
                    bet_this_round=p.bet_this_round, total_invested=p.total_invested,
                    folded=p.folded, all_in=p.all_in,
                ))
        if you is None:
            raise ValueError(f"no player with id {player_id!r}")
        if you.hole_cards is None:
            raise ValueError(f"player {player_id} has no hole cards")

        to_call = max(0.0, self.current_bet - you.bet_this_round)
        # Min raise = current bet + last raise size (or BB if no raise yet)
        min_raise_inc = max(self.last_raise_size, self.blinds[1])
        min_raise_total = (self.current_bet + min_raise_inc) - you.bet_this_round
        min_raise_total = min(min_raise_total, you.stack)

        return HandView(
            your_id=you.id,
            your_hole=you.hole_cards,
            your_stack=you.stack,
            your_position=you.position,
            your_bet_this_round=you.bet_this_round,
            board=tuple(self.board),
            street=self.street,
            pot=self.pot,
            current_bet=self.current_bet,
            to_call=to_call,
            min_raise=min_raise_total,
            others=tuple(others),
            action_history=tuple(self.actions),
        )


# ─── Position assignment ──────────────────────────────────────────────────────

# Position order for N players, starting from the SB. UTG acts first preflop;
# SB acts first postflop (when SB hasn't folded).
_POSITION_ORDERS: dict[int, list[Position]] = {
    2: [Position.SB, Position.BB],                              # heads-up: BTN==SB
    3: [Position.SB, Position.BB, Position.BTN],
    4: [Position.SB, Position.BB, Position.CO, Position.BTN],
    5: [Position.SB, Position.BB, Position.UTG, Position.CO, Position.BTN],
    6: [Position.SB, Position.BB, Position.UTG, Position.MP, Position.CO, Position.BTN],
}


def assign_positions(players: list[Player], button_index: int) -> None:
    """Set each player's position based on the button location."""
    n = len(players)
    if n not in _POSITION_ORDERS:
        raise ValueError(f"unsupported player count: {n} (need 2-6)")
    order = _POSITION_ORDERS[n]
    # In 2-handed, button == SB.
    sb_offset = 0 if n == 2 else 1
    sb_index = (button_index + sb_offset) % n
    for i in range(n):
        seat_idx = (sb_index + i) % n
        players[seat_idx].position = order[i]


# ─── Hand lifecycle ───────────────────────────────────────────────────────────

def start_hand(
    players: list[Player],
    button_index: int,
    deck: Deck,
    blinds: tuple[float, float],
) -> HandState:
    """Set up a fresh hand: assign positions, post blinds, deal hole cards,
    set actor for first preflop action.
    """
    if len(players) < 2:
        raise ValueError("need at least 2 players")

    # Reset per-hand player state
    for p in players:
        p.hole_cards = None
        p.folded = False
        p.all_in = False
        p.bet_this_round = 0.0
        p.total_invested = 0.0

    assign_positions(players, button_index)

    state = HandState(
        players=players,
        button_index=button_index,
        deck=deck,
        blinds=blinds,
    )

    sb_amount, bb_amount = blinds
    sb_player = _find_position(state, Position.SB)
    bb_player = _find_position(state, Position.BB)

    _post_blind(state, sb_player, sb_amount)
    _post_blind(state, bb_player, bb_amount)
    state.current_bet = bb_amount
    state.last_raise_size = bb_amount  # so min-raise = 2*BB

    # Everyone (including blind posters) must voluntarily act preflop. The blind
    # post itself isn't a "real" action.
    state.players_to_act = {p.id for p in state.players if p.can_act}

    # Deal hole cards (one at a time, twice — standard dealing order doesn't
    # affect Monte Carlo correctness but matches table convention)
    for _ in range(2):
        for p in players:
            card = deck.deal_one()
            if p.hole_cards is None:
                p.hole_cards = (card,)  # type: ignore[assignment]
            else:
                p.hole_cards = (p.hole_cards[0], card)  # type: ignore[assignment]

    # Preflop action starts UTG (or, in 2-handed, with the SB / button)
    state.actor_index = _first_to_act_preflop(state)
    return state


def _post_blind(state: HandState, player: Player, amount: float) -> None:
    """Player posts a blind. Goes all-in if they don't have enough."""
    contribution = min(amount, player.stack)
    player.stack -= contribution
    player.bet_this_round += contribution
    player.total_invested += contribution
    state.pot += contribution
    if player.stack == 0:
        player.all_in = True


def _find_position(state: HandState, pos: Position) -> Player:
    for p in state.players:
        if p.position == pos:
            return p
    raise ValueError(f"no player at position {pos}")


def _first_to_act_preflop(state: HandState) -> int:
    """First preflop actor is left of BB (UTG in 6-max, SB in 2-handed)."""
    if len(state.players) == 2:
        # 2-handed: SB acts first preflop (and is the button)
        return state.players.index(_find_position(state, Position.SB))
    bb_idx = state.players.index(_find_position(state, Position.BB))
    return _next_acting_seat(state, bb_idx)


def _first_to_act_postflop(state: HandState) -> int:
    """First postflop actor is left of button (SB if still in)."""
    return _next_acting_seat(state, state.button_index)


def _next_acting_seat(state: HandState, from_index: int) -> int:
    """Find the next seat that can_act, starting AFTER from_index."""
    n = len(state.players)
    for i in range(1, n + 1):
        idx = (from_index + i) % n
        if state.players[idx].can_act:
            return idx
    return from_index  # fallback (shouldn't happen if hand is live)


# ─── Action application ───────────────────────────────────────────────────────

def legal_actions(state: HandState) -> list[ActionType]:
    """What action types is the current actor allowed to take?"""
    p = state.actor()
    actions = [ActionType.FOLD]
    if p.bet_this_round == state.current_bet:
        actions.append(ActionType.CHECK)
    else:
        actions.append(ActionType.CALL)
    if p.stack > 0:
        if state.current_bet == 0:
            actions.append(ActionType.BET)
        else:
            # Can raise only if there are chips left to put in beyond a call
            to_call = state.current_bet - p.bet_this_round
            if p.stack > to_call:
                actions.append(ActionType.RAISE)
    return actions


def apply_action(state: HandState, action: Action) -> ActionRecord:
    """Apply an action by the current actor. Mutates state. Returns a record."""
    p = state.actor()
    legal = legal_actions(state)
    if action.type not in legal:
        raise ValueError(f"action {action.type} not in legal actions {legal}")

    pot_before = state.pot
    is_aggressive = False  # bet or raise → resets players_to_act

    if action.type == ActionType.FOLD:
        p.folded = True

    elif action.type == ActionType.CHECK:
        pass  # no chip movement

    elif action.type == ActionType.CALL:
        to_call = state.current_bet - p.bet_this_round
        contribution = min(to_call, p.stack)
        _commit(state, p, contribution)

    elif action.type == ActionType.BET:
        if action.amount <= 0:
            raise ValueError("bet amount must be positive")
        contribution = min(action.amount, p.stack)
        if contribution < state.blinds[1] and p.stack > state.blinds[1]:
            raise ValueError(
                f"bet must be at least the big blind ({state.blinds[1]}); "
                f"got {contribution}"
            )
        _commit(state, p, contribution)
        state.current_bet = p.bet_this_round
        state.last_raise_size = contribution
        is_aggressive = True

    elif action.type == ActionType.RAISE:
        # `amount` is total chips this player puts in (not raise-to total)
        if action.amount <= 0:
            raise ValueError("raise amount must be positive")
        contribution = min(action.amount, p.stack)
        new_bet_total = p.bet_this_round + contribution
        raise_increment = new_bet_total - state.current_bet
        min_inc = max(state.last_raise_size, state.blinds[1])
        # Allow under-min raise only when going all-in
        going_all_in = (contribution == p.stack)
        if raise_increment < min_inc and not going_all_in:
            raise ValueError(
                f"raise increment {raise_increment} below minimum {min_inc}"
            )
        _commit(state, p, contribution)
        state.current_bet = new_bet_total
        if raise_increment >= min_inc:
            state.last_raise_size = raise_increment
        is_aggressive = True

    record = ActionRecord(
        player_id=p.id,
        action=action,
        street=state.street,
        pot_before=pot_before,
        pot_after=state.pot,
    )
    state.actions.append(record)

    # Update players_to_act
    state.players_to_act.discard(p.id)
    if is_aggressive:
        # Aggression reopens action for everyone else who can still act
        state.players_to_act = {
            q.id for q in state.players if q.can_act and q.id != p.id
        }

    _advance(state)
    return record


def _commit(state: HandState, p: Player, amount: float) -> None:
    """Move chips from player stack to pot (and update player bet trackers)."""
    if amount > p.stack:
        amount = p.stack
    p.stack -= amount
    p.bet_this_round += amount
    p.total_invested += amount
    state.pot += amount
    if p.stack == 0:
        p.all_in = True


def _advance(state: HandState) -> None:
    """Decide what to do after an action: next actor, next street, or end."""
    # Hand ends if only one player remains
    if state.num_in_hand <= 1:
        state.street = Street.SHOWDOWN
        return

    # Round complete if no one still needs to act
    if not state.players_to_act:
        _complete_betting_round(state)
        return

    # Otherwise advance to next player who still needs to act
    state.actor_index = _next_acting_seat(state, state.actor_index)


def _complete_betting_round(state: HandState) -> None:
    """Reset bet_this_round, deal next street, set first actor."""
    for p in state.players:
        p.bet_this_round = 0.0
    state.current_bet = 0.0
    state.last_raise_size = state.blinds[1]

    if state.street == Street.PREFLOP:
        state.street = Street.FLOP
        state.board.extend(state.deck.deal(3))
    elif state.street == Street.FLOP:
        state.street = Street.TURN
        state.board.extend(state.deck.deal(1))
    elif state.street == Street.TURN:
        state.street = Street.RIVER
        state.board.extend(state.deck.deal(1))
    elif state.street == Street.RIVER:
        state.street = Street.SHOWDOWN
        return
    else:
        return

    # If at most one player can still act, no more betting — deal remaining
    # streets and skip to showdown.
    can_act_count = sum(1 for p in state.players if p.can_act)
    if can_act_count <= 1:
        _complete_betting_round(state)  # recurse through remaining streets
        return

    # Set up next round
    state.players_to_act = {p.id for p in state.players if p.can_act}
    state.actor_index = _first_to_act_postflop(state)
    if not state.players[state.actor_index].can_act:
        state.actor_index = _next_acting_seat(state, state.actor_index)


# ─── Showdown and pot awarding ────────────────────────────────────────────────

@dataclass(frozen=True)
class PotAward:
    """Result of awarding one pot (main or side)."""
    amount: float
    eligible_player_ids: tuple[str, ...]
    winner_ids: tuple[str, ...]    # could be multiple on a tie
    per_winner: float


def award_pots(state: HandState) -> list[PotAward]:
    """Compute side pots and award each to the best hand among eligible players.

    Eligibility for each pot is based on `total_invested`: a player who went
    all-in for $20 is only eligible for the portion of the pot up to $20×N
    (where N is the number of players who matched at least that amount).
    """
    if state.num_in_hand == 0:
        return []

    # If only one player remains, they win everything (regardless of cards)
    if state.num_in_hand == 1:
        winner = next(p for p in state.players if p.in_hand)
        award = PotAward(
            amount=state.pot,
            eligible_player_ids=(winner.id,),
            winner_ids=(winner.id,),
            per_winner=state.pot,
        )
        winner.stack += state.pot
        state.pot = 0.0
        return [award]

    # Multi-way showdown: build side pots from total_invested levels.
    # Need 5-card board for evaluation; if not present, deal remaining (this
    # covers the case where everyone went all-in earlier).
    while len(state.board) < 5:
        state.board.extend(state.deck.deal(1))

    in_hand_players = [p for p in state.players if p.in_hand]

    # Build pot layers from each distinct total_invested level among in-hand players
    investments = sorted({p.total_invested for p in state.players if p.total_invested > 0})
    awards: list[PotAward] = []
    prev_level = 0.0
    for level in investments:
        layer_per_player = level - prev_level
        # Contributors: anyone (in or folded) who invested at least `level`
        contributors = [p for p in state.players if p.total_invested >= level]
        layer_amount = layer_per_player * len(contributors)
        # Eligible to win this layer: in-hand players who invested at least `level`
        eligible = [p for p in in_hand_players if p.total_invested >= level]
        if not eligible or layer_amount == 0:
            prev_level = level
            continue

        # Find best hand(s) among eligible
        from .evaluator import evaluate as _eval
        ranked: list[tuple[Player, int]] = []
        for p in eligible:
            assert p.hole_cards is not None
            hs = _eval(p.hole_cards, state.board)
            ranked.append((p, hs.rank))
        best_rank = min(r for _, r in ranked)
        winners = [p for p, r in ranked if r == best_rank]
        per_winner = layer_amount / len(winners)
        for w in winners:
            w.stack += per_winner

        awards.append(PotAward(
            amount=layer_amount,
            eligible_player_ids=tuple(p.id for p in eligible),
            winner_ids=tuple(w.id for w in winners),
            per_winner=per_winner,
        ))
        prev_level = level

    state.pot = 0.0
    return awards
