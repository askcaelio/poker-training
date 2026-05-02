# poker-training

A Texas Hold'em training tool focused on building **probability intuition**, not just playing hands.

> **Status:** Phase 1 (math engine) — works end-to-end on the CLI. Multi-player
> game flow, opponent ranges, and a GUI are on the roadmap below.

## What it teaches

- **Outs counting** — every card that improves your hand, with draw labels (Flush draw, OESD, Gutshot, overcards)
- **Suit-specific probabilities** — "P(diamond on turn) = 9/47 = 19.1%" with the rule-of-4/2 heuristic shown alongside
- **Equity** — Monte Carlo win/tie/lose vs random opponents (validated against textbook matchups)
- **Pot odds & EV** — required equity vs actual equity, with a +EV / -EV / Break-even verdict and the dollar EV of the call
- **Heads-up dashboard** — all of the above bundled into one snapshot per decision

## Sample output

```
════════════════════════════════════════════════════════════
 Flop    Opponents: 1
 Hero:  A♦ K♦
 Board: 7♦ Q♦ 2♣
════════════════════════════════════════════════════════════
 Made hand:    High Card
 Draws:        Flush draw, 2 overcards
 Outs:         23 of 47 unseen   P(any improvement next: 48.9%, by river: 74.5%)
 Rule of 4/2:  92% (heuristic)
 Flush draw ♦:  9 diamonds left  P(next: 19.1%, by river: 35.0%)
 Equity:       win 72.7%  tie 0.7%  lose 26.6%  → 73.0% (10,000 iters)
════════════════════════════════════════════════════════════
 Decision:     pot $100.00, call $50.00 (2:1)
 Need:         33.3%   Have: 73.0%   Edge: +39.7pp
 EV:           +$59.57   →   +EV call
════════════════════════════════════════════════════════════
```

## Layout

```
src/poker/
  cards.py       # Card, Deck, Rank, Suit
  evaluator.py   # 7-card hand strength (wraps treys)
  equity.py      # Monte Carlo equity vs random hands or specific holes
  outs.py        # outs counter, suit-specific draw probabilities
  pot_odds.py    # pot odds, required equity, EV, +EV/-EV verdict
  dashboard.py   # assembles the heads-up display
scripts/
  demo_diamonds.py   # the classic flush-draw probability question
  walkthrough.py     # full hand street-by-street with the dashboard each street
tests/             # pytest, validates against published equity values
```

## Setup

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for env management.

```bash
uv sync
uv run pytest                          # 125 tests
uv run python scripts/walkthrough.py   # see the dashboard in action
```

## Roadmap

**Phase 1 — math engine (current)** — done
**Phase 2 — game flow** — multi-player loop with persistent stacks, betting rounds, side pots, ability to introduce new players mid-session, and bot personalities (TAG / LAG / nit / maniac / station) with explicit bluff frequencies. Goal: actually playable as a long-form trainer.
**Phase 3 — ranges** — 13×13 hand-grid range model, range narrowing as villains act, equity vs ranges (not just random hands). The bridge from "math problem" to "real poker thinking."
**Phase 4 — GUI** — likely SwiftUI for the iOS path. Pygame considered for prototyping only.

## Design notes

- **Engine is UI-agnostic** — `Dashboard`, `DrawAnalysis`, `Equity`, `PotOddsAnalysis` are clean dataclasses ready to feed any frontend.
- **Outs are category changes**, not just kicker bumps. "Pair from high card" counts; "better kicker on the same pair" doesn't.
- **Equity is Monte Carlo** at 10–30k iterations (≈1% precision). Validated against well-known matchups: AA vs KK ≈ 81/19, AKs vs 22 ≈ 50/50, AKo vs QQ ≈ 43/57.
- **Bot personalities over GTO** — for a *trainer*, learning to read player types beats learning to play unexploitable. Solvers are a separate problem.

## Acknowledgments

Built on [treys](https://github.com/ihendley/treys) for the underlying 7-card hand evaluator.
