"""Monte Carlo equity calculator.

Equity = probability of winning a showdown given current information. We deal
random cards for the unknowns (opponent hole cards if not specified, plus any
remaining board cards) and tally how often the hero wins or ties.

Two entry points:
  - equity_vs_random: opponents get random hole cards
  - equity_vs_hands:  opponents have specific known hole cards

Both return an Equity object with win/tie/lose percentages (0-100 range).
Standard convention: a tie counts as 1/n_players for win-rate purposes; we
expose both raw counts and pct so the caller can decide how to display.

Performance: the inner loop calls treys directly (skipping our HandStrength
wrapper) and converts cards to treys ints once at the top. That's ~5-7x
faster than going through evaluate() per-iteration, which matters when bots
hammer this function during simulations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from treys import Card as TreysCard
from treys import Evaluator as TreysEvaluator

from .cards import Card, full_deck

_TREYS_EVAL = TreysEvaluator()
_TREYS_EVALUATE = _TREYS_EVAL.evaluate  # bound method, faster lookup in hot loop


@dataclass(frozen=True)
class Equity:
    """Result of an equity calculation.

    win_pct + tie_pct + lose_pct == 100 (within float precision).
    equity_pct is the standard "share of pot" measure: wins + (ties / n_players).
    """

    win_pct: float
    tie_pct: float
    lose_pct: float
    equity_pct: float
    iterations: int

    def __str__(self) -> str:
        return (
            f"win {self.win_pct:.1f}%  tie {self.tie_pct:.1f}%  "
            f"lose {self.lose_pct:.1f}%  equity {self.equity_pct:.1f}% "
            f"({self.iterations:,} iters)"
        )


def _validate_no_duplicates(*card_lists: Iterable[Card]) -> set[Card]:
    """Flatten card lists, ensure no duplicates, return as a set."""
    seen: set[Card] = set()
    for cards in card_lists:
        for c in cards:
            if c in seen:
                raise ValueError(f"duplicate card: {c}")
            seen.add(c)
    return seen


def _to_treys(cards: Iterable[Card]) -> list[int]:
    return [TreysCard.new(str(c)) for c in cards]


def equity_vs_random(
    hole: Iterable[Card],
    board: Iterable[Card] = (),
    num_opponents: int = 1,
    iterations: int = 10_000,
    rng: random.Random | None = None,
) -> Equity:
    """Hero's equity against `num_opponents` opponents holding random hands.

    `board` may be 0, 3, 4, or 5 cards (preflop / flop / turn / river).
    Iterations of 10k give ~1% precision; 50k gives ~0.5%.
    """
    if num_opponents < 1:
        raise ValueError("need at least 1 opponent")

    hole = list(hole)
    board = list(board)
    if len(hole) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hole)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")

    known = _validate_no_duplicates(hole, board)
    rng = rng if rng is not None else random.Random()

    # Convert everything to treys ints up front; the inner loop is pure ints.
    treys_hole = _to_treys(hole)
    treys_board = _to_treys(board)
    treys_deck = _to_treys(c for c in full_deck() if c not in known)

    board_to_draw = 5 - len(board)
    draw_count = board_to_draw + 2 * num_opponents
    sample = rng.sample
    evaluate_fn = _TREYS_EVALUATE

    wins = ties = losses = 0
    for _ in range(iterations):
        drawn = sample(treys_deck, draw_count)
        runout = treys_board + drawn[:board_to_draw]

        hero_rank = evaluate_fn(runout, treys_hole)
        outcome = 0  # 0=win, 1=tie, 2=lose
        opp_idx = board_to_draw
        for _opp in range(num_opponents):
            opp_rank = evaluate_fn(runout, drawn[opp_idx : opp_idx + 2])
            opp_idx += 2
            if opp_rank < hero_rank:  # treys: lower = stronger
                outcome = 2
                break
            if opp_rank == hero_rank:
                outcome = 1

        if outcome == 0:
            wins += 1
        elif outcome == 1:
            ties += 1
        else:
            losses += 1

    return _build_equity(wins, ties, losses, num_opponents + 1, iterations)


def equity_vs_hands(
    hole: Iterable[Card],
    opponents_hole: Iterable[Iterable[Card]],
    board: Iterable[Card] = (),
    iterations: int = 10_000,
    rng: random.Random | None = None,
) -> Equity:
    """Hero's equity against opponents with KNOWN hole cards.

    Useful for "what if villain has exactly AK?" analysis. Only the remaining
    board cards are randomized.
    """
    hole = list(hole)
    board = list(board)
    opps = [list(o) for o in opponents_hole]

    if len(hole) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hole)}")
    for i, o in enumerate(opps):
        if len(o) != 2:
            raise ValueError(f"opponent {i} needs exactly 2 hole cards, got {len(o)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")

    known = _validate_no_duplicates(hole, board, *opps)
    rng = rng if rng is not None else random.Random()

    treys_hole = _to_treys(hole)
    treys_board = _to_treys(board)
    treys_opps = [_to_treys(o) for o in opps]
    treys_deck = _to_treys(c for c in full_deck() if c not in known)

    board_to_draw = 5 - len(board)
    n_players = 1 + len(opps)

    # River already → deterministic, single iteration.
    if board_to_draw == 0:
        iterations = 1

    sample = rng.sample
    evaluate_fn = _TREYS_EVALUATE

    wins = ties = losses = 0
    for _ in range(iterations):
        drawn = sample(treys_deck, board_to_draw) if board_to_draw else []
        runout = treys_board + drawn

        hero_rank = evaluate_fn(runout, treys_hole)
        outcome = 0
        for opp_hole in treys_opps:
            opp_rank = evaluate_fn(runout, opp_hole)
            if opp_rank < hero_rank:
                outcome = 2
                break
            if opp_rank == hero_rank:
                outcome = 1

        if outcome == 0:
            wins += 1
        elif outcome == 1:
            ties += 1
        else:
            losses += 1

    return _build_equity(wins, ties, losses, n_players, iterations)


def _build_equity(wins: int, ties: int, losses: int, n_players: int, iterations: int) -> Equity:
    win_pct = 100.0 * wins / iterations
    tie_pct = 100.0 * ties / iterations
    lose_pct = 100.0 * losses / iterations
    equity_pct = win_pct + tie_pct / n_players
    return Equity(
        win_pct=win_pct,
        tie_pct=tie_pct,
        lose_pct=lose_pct,
        equity_pct=equity_pct,
        iterations=iterations,
    )
