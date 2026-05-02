"""Rendering functions shared by the static mockup and the live Flask app.

Pure functions: take game state + computed snapshot, return HTML strings.
No Flask dependency here — that's in app.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from poker.cards import Card, Suit
from poker.game import HandState, Player, Position


HUMAN_ID = "You"


# ─── Card rendering ───────────────────────────────────────────────────────────

def card_html(card: Card, *, face_down: bool = False, classes: str = "") -> str:
    if face_down:
        return f'<div class="card card-back {classes}"><div class="card-back-pattern"></div></div>'

    suit_glyph = card.suit.glyph
    rank_str = card.rank.char
    suit_class = "suit-red" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "suit-black"
    return (
        f'<div class="card {suit_class} {classes}">'
        f'<div class="card-corner top"><span class="card-rank">{rank_str}</span><span class="card-suit">{suit_glyph}</span></div>'
        f'<div class="card-pip">{suit_glyph}</div>'
        f'<div class="card-corner bottom"><span class="card-rank">{rank_str}</span><span class="card-suit">{suit_glyph}</span></div>'
        '</div>'
    )


def explainer(title: str, body: str, anchor: str) -> str:
    return (
        f'<details class="explainer" id="exp-{anchor}">'
        f'<summary><span class="explainer-icon">?</span><span class="explainer-label">What is {title}?</span></summary>'
        f'<div class="explainer-body">{body}</div>'
        '</details>'
    )


# ─── Table rendering ──────────────────────────────────────────────────────────

# Hero is always seat 0; we display the table with hero at the bottom and bots
# arranged around. We use a position-style mapping based on the offset from
# hero, not on absolute seat indices.
_RING_POSITIONS = ["bottom", "bottom-left", "left", "top-left", "top", "top-right", "right", "bottom-right"]


def _seat_html_at(state: HandState, seat_idx: int, *, is_hero: bool) -> str:
    p = state.players[seat_idx]
    cards_html = ""
    if is_hero and p.hole_cards:
        cards_html = "".join(card_html(c, classes="seat-card") for c in p.hole_cards)
    elif p.hole_cards and not p.folded:
        cards_html = card_html(p.hole_cards[0], face_down=True, classes="mini") + \
                     card_html(p.hole_cards[1], face_down=True, classes="mini")

    pos_label = p.position.value if p.position else "—"
    folded_class = "folded" if p.folded else ""
    acting_class = "acting" if (not state.is_complete and seat_idx == state.actor_index) else ""
    hero_class = "hero-seat" if is_hero else ""

    folded_text = '<span class="seat-name">folded</span>' if p.folded else ''
    bet_text = f'<div class="bet-this-round">${p.bet_this_round:.2f}</div>' if p.bet_this_round > 0 else ''

    return (
        f'<div class="seat {folded_class} {acting_class} {hero_class}">'
        f'<div class="seat-cards">{cards_html}</div>'
        f'<div class="seat-info"><span class="seat-pos">{pos_label}</span><span class="seat-id">{p.id}</span></div>'
        f'<div class="stack">${p.stack:.2f}</div>'
        f'{folded_text}'
        f'{bet_text}'
        '</div>'
    )


def render_table(state: HandState) -> str:
    """Render the poker table panel (left side of dashboard)."""
    n = len(state.players)
    hero_seat = 0
    # Build seat HTMLs in a clockwise list starting from hero, then assign
    # to ring positions: hero=bottom, others around.
    seats_html = []
    ring_classes = {
        0: "ring-bottom",          # hero
        1: "ring-bottom-right",    # next clockwise
        2: "ring-right",
        3: "ring-top",             # opposite
        4: "ring-left",
        5: "ring-bottom-left",
    }
    for offset in range(n):
        seat_idx = (hero_seat + offset) % n
        position_class = ring_classes.get(offset, "")
        seat = _seat_html_at(state, seat_idx, is_hero=(offset == 0))
        # Wrap with position class
        seat = seat.replace('class="seat ', f'class="seat {position_class} ', 1)
        seats_html.append(seat)

    board_html = "".join(card_html(c, classes="board-card") for c in state.board) if state.board else \
                 '<div class="board-empty">— preflop —</div>'

    return (
        '<div class="table-frame">'
        + "".join(seats_html)
        + f'<div class="pot-display">POT<span class="amount">${state.pot:.2f}</span></div>'
        + f'<div class="board">{board_html}</div>'
        + '</div>'
    )


# ─── Action log ───────────────────────────────────────────────────────────────

def render_action_log(state: HandState, max_entries: int = 14) -> str:
    if not state.actions:
        return '<div class="log-empty">No actions yet</div>'
    entries = state.actions[-max_entries:]
    rows = []
    for rec in entries:
        amt = ""
        if rec.action.amount > 0:
            amt = f'<span class="log-amt">${rec.action.amount:.2f}</span>'
        is_human = rec.player_id == HUMAN_ID
        you_class = "you" if is_human else ""
        rows.append(
            f'<div class="log-row {you_class}">'
            f'<span class="log-street">{rec.street.value[:3].upper()}</span>'
            f'<span class="log-id">{rec.player_id}</span>'
            f'<span class="log-action">{rec.action.type.value}</span>'
            f'{amt}'
            '</div>'
        )
    return '<div class="action-log">' + "".join(rows) + '</div>'


# ─── Awards / showdown ────────────────────────────────────────────────────────

def render_showdown(state: HandState, awards: list, hand_ended_via_fold: bool) -> str:
    if not awards:
        return ""

    if hand_ended_via_fold:
        # Single award, no need to show cards
        winner_ids = ", ".join(awards[0].winner_ids)
        amount = awards[0].amount
        return (
            '<div class="showdown">'
            '<div class="showdown-title">HAND OVER</div>'
            f'<div class="showdown-result"><strong>{winner_ids}</strong> wins ${amount:.2f} (uncontested)</div>'
            '</div>'
        )

    # Showdown — show all in-hand players' cards
    cards_html = []
    for p in state.players:
        if p.in_hand and p.hole_cards:
            cards = "".join(card_html(c, classes="showdown-card") for c in p.hole_cards)
            won_class = "won" if any(p.id in a.winner_ids for a in awards) else ""
            won_amt = sum(a.per_winner for a in awards if p.id in a.winner_ids)
            won_label = f' <span class="won-amt">+${won_amt:.2f}</span>' if won_amt > 0 else ''
            cards_html.append(
                f'<div class="showdown-player {won_class}">'
                f'<div class="showdown-name">{p.id}</div>'
                f'<div class="showdown-cards">{cards}</div>'
                f'{won_label}'
                '</div>'
            )

    pot_lines = []
    for a in awards:
        winners = ", ".join(a.winner_ids)
        pot_lines.append(f'<div class="pot-line">${a.amount:.2f} → {winners}</div>')

    return (
        '<div class="showdown">'
        '<div class="showdown-title">SHOWDOWN</div>'
        '<div class="showdown-players">' + "".join(cards_html) + '</div>'
        '<div class="showdown-pots">' + "".join(pot_lines) + '</div>'
        '</div>'
    )
