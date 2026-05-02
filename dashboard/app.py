"""Flask app for the interactive dashboard.

Run with:
    uv run python -m dashboard.app
    open http://127.0.0.1:5000
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from flask import Flask, request

from poker.agents import (
    Agent, BotAgent, make_lag, make_maniac, make_nit, make_station, make_tag,
)
from poker.cards import Deck
from poker.game import (
    Action, ActionType, HandState, Player, Position, Street,
    apply_action, award_pots, start_hand,
)
from poker.opponent_model import OpponentTable
from poker.range_tracker import RangeTracker

from .brain import HUMAN_ID, render_brain
from .page import render_page
from .render import render_action_log, render_showdown, render_table


app = Flask(__name__)


# ─── Game session state (single global, single user for MVP) ─────────────────

@dataclass
class GameSession:
    bots: dict[str, BotAgent] = field(default_factory=dict)
    bot_order: list[Agent] = field(default_factory=list)  # in seat order with placeholder for hero
    opponent_table: OpponentTable = field(default_factory=OpponentTable)
    range_tracker: RangeTracker = field(default_factory=RangeTracker)
    button_index: int = 0
    stacks: list[float] = field(default_factory=list)
    hand_state: Optional[HandState] = None
    hand_id: int = 0
    last_awards: list = field(default_factory=list)
    last_hand_via_fold: bool = False
    rng: random.Random = field(default_factory=lambda: random.Random())


_session: Optional[GameSession] = None


def get_session() -> GameSession:
    global _session
    if _session is None:
        _session = _new_session()
    return _session


def _new_session() -> GameSession:
    sess = GameSession()
    sess.rng = random.Random()
    # Hero is seat 0; 5 bot opponents around the table
    sess.bots = {
        "Nit": make_nit("Nit"),
        "TAG_2": make_tag("TAG_2"),
        "LAG": make_lag("LAG"),
        "Maniac": make_maniac("Maniac"),
        "Station": make_station("Station"),
    }
    # Wire opponent table + range tracker to bots that adapt
    for bot in sess.bots.values():
        bot.opponent_table = sess.opponent_table
        bot.range_tracker = sess.range_tracker

    # Seat order: You (seat 0) + 5 bots clockwise
    bot_names_in_order = ["Nit", "TAG_2", "LAG", "Maniac", "Station"]
    sess.bot_order = [None] + [sess.bots[name] for name in bot_names_in_order]
    sess.stacks = [100.0] * 6
    sess.button_index = 0
    return sess


def _start_new_hand(sess: GameSession) -> None:
    """Set up a fresh hand and advance bots until the human's turn (or hand ends)."""
    # Auto-rebuy busted players
    sess.stacks = [s if s >= 1.0 else 100.0 for s in sess.stacks]

    # Build players in seat order; index 0 is the human
    players = [Player(id=HUMAN_ID, stack=sess.stacks[0])]
    for bot in sess.bot_order[1:]:
        players.append(Player(id=bot.name, stack=sess.stacks[len(players)]))

    deck = Deck(rng=sess.rng)
    sess.hand_state = start_hand(players, sess.button_index, deck, blinds=(0.5, 1.0))
    sess.hand_id += 1
    sess.last_awards = []
    sess.last_hand_via_fold = False

    # Initialize the range tracker for this hand (uses observed stats to classify)
    sess.range_tracker.init_hand(
        opponent_ids=[p.id for p in players if p.id != HUMAN_ID],
        opponent_table=sess.opponent_table,
    )

    _advance_bots_until_human(sess)


def _advance_bots_until_human(sess: GameSession) -> None:
    """Run bot decisions until the human is to act, or the hand ends."""
    state = sess.hand_state
    if state is None:
        return

    safety = 200
    while not state.is_complete and safety > 0:
        safety -= 1
        actor = state.actor()

        if actor.id == HUMAN_ID and actor.can_act:
            return  # wait for human

        if not actor.can_act:
            # Shouldn't happen — engine handles this — but guard
            break

        # Bot's turn
        bot = sess.bots.get(actor.id)
        if bot is None:
            apply_action(state, Action(ActionType.FOLD))
            continue

        view = state.view_for(actor.id)

        # Snapshot context for range narrowing
        is_facing_raise = (
            view.current_bet > sess.hand_state.blinds[1]
            if view.street == Street.PREFLOP
            else view.current_bet > 0
        )
        is_threebet = (
            view.street == Street.PREFLOP
            and view.current_bet > sess.hand_state.blinds[1] * 1.5
        )

        action = bot.decide(view)
        try:
            apply_action(state, action)
        except ValueError:
            apply_action(state, Action(ActionType.FOLD))

        # Narrow that bot's range based on the action just taken
        applied = state.actions[-1]
        sess.range_tracker.narrow_for_action(
            opponent_id=applied.player_id,
            action_type=applied.action.type.value,
            street=applied.street.value,
            is_facing_raise=is_facing_raise,
            is_threebet_situation=is_threebet,
            board=list(state.board),
        )

    # Hand complete — award pots and update opponent stats
    if state.is_complete:
        _finalize_hand(sess)


def _finalize_hand(sess: GameSession) -> None:
    state = sess.hand_state
    if state is None:
        return
    sess.last_hand_via_fold = state.num_in_hand <= 1
    sess.last_awards = award_pots(state)
    # Persist final stacks
    sess.stacks = [p.stack for p in state.players]
    # Rotate button for next hand
    sess.button_index = (sess.button_index + 1) % len(state.players)
    # Update opponent table from this hand
    record = {
        "players": [{"id": p.id} for p in state.players],
        "actions": [
            {
                "player_id": rec.player_id,
                "action_type": rec.action.type.value,
                "amount": rec.action.amount,
                "street": rec.street.value,
            }
            for rec in state.actions
        ],
    }
    sess.opponent_table.observe_hand(record)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    sess = get_session()
    if sess.hand_state is None:
        _start_new_hand(sess)
    return _render(sess)


@app.route("/act", methods=["POST"])
def act():
    sess = get_session()
    if sess.hand_state is None or sess.hand_state.is_complete:
        return _render(sess)

    action_type = request.form.get("action_type", "fold")
    try:
        amount = float(request.form.get("amount", 0))
    except ValueError:
        amount = 0.0

    state = sess.hand_state
    actor = state.actor()
    if actor.id != HUMAN_ID:
        # Not your turn
        return _render(sess)

    # Snapshot for narrowing
    view = state.view_for(HUMAN_ID)
    is_facing_raise = (
        view.current_bet > state.blinds[1]
        if view.street == Street.PREFLOP
        else view.current_bet > 0
    )
    is_threebet = (
        view.street == Street.PREFLOP
        and view.current_bet > state.blinds[1] * 1.5
    )

    try:
        action = Action(ActionType(action_type), amount=amount)
        apply_action(state, action)
    except (ValueError, KeyError):
        # Invalid input — ignore and just re-render
        return _render(sess)

    # Narrow human's range from observers' POV (not strictly needed but
    # consistent with bot tracking)
    applied = state.actions[-1]
    sess.range_tracker.narrow_for_action(
        opponent_id=applied.player_id,
        action_type=applied.action.type.value,
        street=applied.street.value,
        is_facing_raise=is_facing_raise,
        is_threebet_situation=is_threebet,
        board=list(state.board),
    )

    _advance_bots_until_human(sess)
    return _render(sess)


@app.route("/next_hand", methods=["POST"])
def next_hand():
    sess = get_session()
    _start_new_hand(sess)
    return _render(sess)


def _render(sess: GameSession) -> str:
    state = sess.hand_state
    if state is None:
        return "<p>No game state. Visit / to start.</p>"

    table_html = render_table(state)
    log_html = render_action_log(state)
    showdown_html = ""
    if state.is_complete:
        showdown_html = render_showdown(state, sess.last_awards, sess.last_hand_via_fold)
    brain_html = render_brain(state, sess.opponent_table, sess.range_tracker)

    button_id = state.players[sess.button_index].id if sess.button_index < len(state.players) else "?"

    return render_page(
        table_html=table_html,
        brain_html=brain_html,
        log_html=log_html,
        showdown_html=showdown_html,
        hand_id=sess.hand_id,
        button_id=button_id,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
