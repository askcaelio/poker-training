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
from .range_model import Range, sample_combos_from_range

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


def equity_vs_range(
    hole: Iterable[Card],
    villain_range: Range,
    board: Iterable[Card] = (),
    iterations: int = 5_000,
    rng: random.Random | None = None,
) -> Equity:
    """Hero's equity against ONE opponent whose holdings are drawn from
    `villain_range`, weighted by combo count × range weight.

    For each iteration:
      1. Sample a villain hand from the range (weighted)
      2. Deal remaining board cards
      3. Evaluate, count win/tie/lose

    `board` may be 0/3/4/5 cards. Range hands that conflict with hole or board
    cards are excluded from sampling.
    """
    hole = list(hole)
    board = list(board)
    if len(hole) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hole)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")

    known = _validate_no_duplicates(hole, board)
    rng = rng if rng is not None else random.Random()

    # Sample combos from the range, excluding cards in hero/board
    weighted_combos = sample_combos_from_range(villain_range, known)
    if not weighted_combos:
        # Range is empty after excluding known cards (rare but possible)
        return Equity(win_pct=0, tie_pct=0, lose_pct=100, equity_pct=0, iterations=0)

    combos, weights = zip(*weighted_combos)
    treys_combos = [_to_treys(c) for c in combos]

    treys_hole = _to_treys(hole)
    treys_board = _to_treys(board)
    treys_deck = _to_treys(c for c in full_deck() if c not in known)

    board_to_draw = 5 - len(board)
    sample = rng.sample
    choices = rng.choices  # weighted sampling
    evaluate_fn = _TREYS_EVALUATE

    wins = ties = losses = 0
    for _ in range(iterations):
        # Pick a villain combo (weighted by combo count × range weight)
        opp_hole = choices(treys_combos, weights=weights, k=1)[0]
        # Build a deck excluding villain's cards too
        excluded = set(treys_hole) | set(treys_board) | set(opp_hole)
        deck_iter = [c for c in treys_deck if c not in excluded]
        runout = treys_board + (sample(deck_iter, board_to_draw) if board_to_draw else [])

        hero_rank = evaluate_fn(runout, treys_hole)
        opp_rank = evaluate_fn(runout, opp_hole)
        if hero_rank < opp_rank:    # treys: lower = stronger
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1
        else:
            losses += 1

    return _build_equity(wins, ties, losses, 2, iterations)


def equity_vs_ranges(
    hole: Iterable[Card],
    opponent_ranges: list[Range],
    board: Iterable[Card] = (),
    iterations: int = 5_000,
    rng: random.Random | None = None,
) -> Equity:
    """Hero's equity vs N opponents, each with their own inferred range.

    For each iter:
      1. Sample one combo per opponent (drawn from their range, weighted by
         combo count × range weight, with no card conflicts).
      2. Deal remaining board cards.
      3. Evaluate hero vs each opponent; tally win/tie/lose.

    Falls back to a tie if all sampling attempts fail (rare; happens when
    ranges are extremely tight + heavily blocker-conflicted).
    """
    hole = list(hole)
    board = list(board)
    if len(hole) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hole)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")
    if not opponent_ranges:
        raise ValueError("need at least one opponent range")

    known = _validate_no_duplicates(hole, board)
    rng = rng if rng is not None else random.Random()

    # Pre-build (combos, weights, treys_combos) per opponent, excluding hero+board cards
    per_opp_combos: list[list[tuple[Card, Card]]] = []
    per_opp_weights: list[list[float]] = []
    per_opp_treys: list[list[list[int]]] = []
    for r in opponent_ranges:
        weighted = sample_combos_from_range(r, known)
        if not weighted:
            # Empty range after blockers — treat as random
            return equity_vs_random(
                hole, board, num_opponents=len(opponent_ranges),
                iterations=iterations, rng=rng,
            )
        combos, weights = zip(*weighted)
        per_opp_combos.append(list(combos))
        per_opp_weights.append(list(weights))
        per_opp_treys.append([_to_treys(c) for c in combos])

    treys_hole = _to_treys(hole)
    treys_board = _to_treys(board)
    treys_full_deck = _to_treys(c for c in full_deck() if c not in known)

    board_to_draw = 5 - len(board)
    sample = rng.sample
    choices = rng.choices
    evaluate_fn = _TREYS_EVALUATE

    wins = ties = losses = 0
    for _ in range(iterations):
        # Sample one combo per opponent without conflicts.
        used: set[int] = set(treys_hole) | set(treys_board)
        opp_holes: list[list[int]] = []
        ok = True
        for i in range(len(opponent_ranges)):
            # Try a few times; if no non-conflicting combo found, give up this iter
            picked = None
            for _try in range(8):
                cand_idx = choices(range(len(per_opp_treys[i])), weights=per_opp_weights[i], k=1)[0]
                cand = per_opp_treys[i][cand_idx]
                if cand[0] not in used and cand[1] not in used:
                    picked = cand
                    break
            if picked is None:
                ok = False
                break
            opp_holes.append(picked)
            used.add(picked[0]); used.add(picked[1])
        if not ok:
            continue

        # Deal remaining board cards (excluding everything used so far)
        remaining_deck = [c for c in treys_full_deck if c not in used]
        runout = treys_board + (sample(remaining_deck, board_to_draw) if board_to_draw else [])

        hero_rank = evaluate_fn(runout, treys_hole)
        outcome = 0  # 0=win, 1=tie, 2=lose
        for opp in opp_holes:
            opp_rank = evaluate_fn(runout, opp)
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

    total = wins + ties + losses
    if total == 0:
        # Pathological — all iterations failed sampling. Fall back to random.
        return equity_vs_random(
            hole, board, num_opponents=len(opponent_ranges),
            iterations=iterations, rng=rng,
        )
    return _build_equity(wins, ties, losses, len(opponent_ranges) + 1, total)


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
