"""Heads-up dashboard — the assembled trainer view for any hand state.

Bundles hand strength, outs/draws, equity, suit-specific probabilities, and
(when a betting decision is involved) pot odds + EV verdict into one object.

Two functions:
  build(...)   produces a Dashboard with all the numbers
  render(...)  formats it for the CLI

The Dashboard dataclass is the contract a future GUI will consume — same
numbers, different presentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .cards import Card, Suit
from .equity import Equity, equity_vs_random
from .evaluator import HandStrength, evaluate
from .outs import DrawAnalysis, analyze_outs, prob_specific_suit
from .pot_odds import PotOddsAnalysis, analyze_call


def _street_name(board_size: int) -> str:
    return {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}[board_size]


@dataclass(frozen=True)
class SuitDraw:
    """P(see at least one more of this suit) for a suit hero is drawing toward."""
    suit: Suit
    visible: int
    remaining: int
    p_next: float
    p_by_river: float


@dataclass(frozen=True)
class Dashboard:
    hole: tuple[Card, ...]
    board: tuple[Card, ...]
    street: str
    num_opponents: int
    hand_strength: HandStrength | None       # None preflop (no 5 cards yet)
    draw_analysis: DrawAnalysis | None       # None preflop or river
    equity: Equity
    suit_draws: tuple[SuitDraw, ...] = field(default_factory=tuple)
    pot_odds: PotOddsAnalysis | None = None  # None when no pending decision


def build(
    hole: Iterable[Card],
    board: Iterable[Card] = (),
    num_opponents: int = 1,
    pot: float | None = None,
    call: float | None = None,
    iterations: int = 10_000,
    rng: random.Random | None = None,
) -> Dashboard:
    """Build a Dashboard for the given hand state.

    `pot` and `call` are optional — provide them to get a pot-odds verdict.
    Both required together; passing one without the other raises.
    """
    hole = tuple(hole)
    board = tuple(board)
    if (pot is None) != (call is None):
        raise ValueError("pot and call must be provided together (or neither)")

    street = _street_name(len(board))

    # Hand strength: only well-defined once we have 5+ cards total.
    hand_strength: HandStrength | None = None
    if len(hole) + len(board) >= 5:
        hand_strength = evaluate(hole, board)

    # Draw analysis: only on flop/turn (3 or 4 board cards).
    draw_analysis: DrawAnalysis | None = None
    if len(board) in (3, 4):
        draw_analysis = analyze_outs(list(hole), list(board))

    # Equity: works at any street.
    equity = equity_vs_random(
        list(hole), list(board), num_opponents=num_opponents,
        iterations=iterations, rng=rng,
    )

    # Suit-specific probabilities for any suit hero is drawing toward
    # (4 visible of one suit = flush draw — the only practical case to surface).
    suit_draws: list[SuitDraw] = []
    if len(board) in (3, 4):
        all_visible = list(hole) + list(board)
        for suit in Suit:
            visible = sum(1 for c in all_visible if c.suit == suit)
            if visible == 4:
                info = prob_specific_suit(list(hole), list(board), suit)
                suit_draws.append(SuitDraw(
                    suit=suit,
                    visible=visible,
                    remaining=info["suit_remaining"],
                    p_next=info["p_next"],
                    p_by_river=info["p_by_river"],
                ))

    # Pot-odds analysis (only when a decision is on the table)
    pot_odds: PotOddsAnalysis | None = None
    if pot is not None and call is not None:
        pot_odds = analyze_call(pot=pot, call=call, equity_pct=equity.equity_pct)

    return Dashboard(
        hole=hole,
        board=board,
        street=street,
        num_opponents=num_opponents,
        hand_strength=hand_strength,
        draw_analysis=draw_analysis,
        equity=equity,
        suit_draws=tuple(suit_draws),
        pot_odds=pot_odds,
    )


def render(d: Dashboard) -> str:
    """Pretty-print a Dashboard for the CLI."""
    lines: list[str] = []
    sep = "═" * 60
    lines.append(sep)
    hole_str = " ".join(c.pretty() for c in d.hole)
    board_str = " ".join(c.pretty() for c in d.board) if d.board else "(none)"
    lines.append(f" {d.street}    Opponents: {d.num_opponents}")
    lines.append(f" Hero:  {hole_str}")
    lines.append(f" Board: {board_str}")
    lines.append(sep)

    # Hand strength
    if d.hand_strength is not None:
        lines.append(f" Made hand:    {d.hand_strength.category}")

    # Draws
    if d.draw_analysis is not None:
        a = d.draw_analysis
        labels = ", ".join(a.labels) if a.labels else "(no draws)"
        lines.append(f" Draws:        {labels}")
        lines.append(
            f" Outs:         {a.out_count} of {a.unseen} unseen   "
            f"P(any improvement next: {a.prob_hit_next * 100:.1f}%, "
            f"by river: {a.prob_hit_by_river * 100:.1f}%)"
        )
        # Breakdown by destination category — much more pedagogical than a single number
        if a.outs_by_destination:
            order = ["Flush", "Straight", "Three of a Kind", "Two Pair", "Pair",
                     "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]
            parts = []
            for cat in order:
                if cat in a.outs_by_destination:
                    parts.append(f"{len(a.outs_by_destination[cat])}→{cat}")
            if parts:
                lines.append(f" Breakdown:    {' | '.join(parts)}")
        rule_note = "" if a.rule_reliable else "  ⚠ unreliable above ~13 outs"
        lines.append(f" Rule of 4/2:  {a.rule_of_4_or_2 * 100:.0f}% (heuristic){rule_note}")

    # Suit-specific (flush draws only)
    for sd in d.suit_draws:
        glyph = sd.suit.glyph
        lines.append(
            f" Flush draw {glyph}:  {sd.remaining} {sd.suit.name.lower()} left  "
            f"P(next: {sd.p_next * 100:.1f}%, by river: {sd.p_by_river * 100:.1f}%)"
        )

    # Equity
    eq = d.equity
    lines.append(
        f" Equity:       win {eq.win_pct:.1f}%  tie {eq.tie_pct:.1f}%  "
        f"lose {eq.lose_pct:.1f}%  → {eq.equity_pct:.1f}% "
        f"({eq.iterations:,} iters)"
    )

    # Pot odds
    if d.pot_odds is not None:
        lines.append(sep)
        po = d.pot_odds
        lines.append(
            f" Decision:     pot ${po.pot_size:.2f}, call ${po.call_amount:.2f} "
            f"({po.pot_odds_ratio})"
        )
        lines.append(
            f" Need:         {po.required_equity_pct:.1f}%   "
            f"Have: {po.actual_equity_pct:.1f}%   "
            f"Edge: {po.edge_pp:+.1f}pp"
        )
        sign = "+" if po.ev >= 0 else ""
        lines.append(f" EV:           {sign}${po.ev:.2f}   →   {po.verdict}")

    lines.append(sep)
    return "\n".join(lines)
