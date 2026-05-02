# poker-training

A Texas Hold'em training tool focused on building **probability intuition** —
and a self-contained simulation engine that runs autonomously for analysis or
ML training data.

> **Status:** Phase 1 (math engine) ✅ + Phase 2 (game flow + bots) ✅ +
> Phase 2.5 (opponent modeling + fold equity) ✅ + Phase 3 (range modeling
> with postflop narrowing + multiway) ✅
> Bet sizing by texture and a GUI are next on the roadmap.

## What it does

**Two modes, one engine:**

- **`scripts/play.py`** — sit at a 6-max table against 5 bot opponents
  with the dashboard shown for every decision you make (pot odds, equity,
  outs, EV verdict). Build intuition by playing real hands.
- **`scripts/simulate.py`** — autonomous bot-vs-bot simulation, dumps every
  action and showdown to JSONL. Use the data for analysis, or train ML models
  on hand histories.

## What it teaches

- **Outs counting** — every card that improves your hand, broken down by
  destination ("9 → Flush, 14 → Pair")
- **Suit-specific probabilities** — "P(diamond on turn) = 9/47 = 19.1%"
  alongside the rule-of-4/2 heuristic, with a warning when the heuristic
  diverges (>13 outs)
- **Equity** — Monte Carlo win/tie/lose vs random opponents (validated
  against textbook matchups: AA vs KK ≈ 81/19, AKs vs 22 ≈ 50/50)
- **Pot odds & EV** — required equity vs actual equity, with a verdict
  (+EV / -EV / Break-even) and the dollar EV of the call
- **Bot reading** — opponents have explicit personalities you can study:
  Nit (~7% VPIP), TAG (~20%), LAG (~30%), Maniac (~50%), Station (~85%)
- **Adaptive bots** — TAG/LAG/Nit track per-opponent stats (VPIP, fold-to-cbet,
  aggression factor) and use *fold equity* in their decisions. They stop
  bluffing stations after ~20 hands of evidence. Maniac/Station deliberately
  do NOT adapt — that's their archetype.

## Sample dashboard

```
════════════════════════════════════════════════════════════
 Flop    Opponents: 1
 Hero:  A♦ K♦
 Board: 7♦ Q♦ 2♣
════════════════════════════════════════════════════════════
 Made hand:    High Card
 Draws:        Flush draw, 2 overcards
 Outs:         23 of 47 unseen   P(any improvement next: 48.9%, by river: 74.5%)
 Breakdown:    9→Flush | 14→Pair
 Rule of 4/2:  92% (heuristic)  ⚠ unreliable above ~13 outs
 Flush draw ♦:  9 diamonds left  P(next: 19.1%, by river: 35.0%)
 Equity:       win 72.7%  tie 0.7%  lose 26.6%  → 73.0% (10,000 iters)
════════════════════════════════════════════════════════════
 Decision:     pot $100.00, call $50.00 (2:1)
 Need:         33.3%   Have: 73.0%   Edge: +39.7pp
 EV:           +$59.57   →   +EV call
════════════════════════════════════════════════════════════
```

## Sample simulation output

500-hand, 6-handed mixed table:

```
Sim: 500 hands

  player          hands   VPIP    PFR   won/hand   BB/100
  ───────────────────────────────────────────────────────
  LAG_1             500  30.0%  21.0%      4.77   +476.6
  Maniac_1          500  55.0%  31.5%     -5.68   -568.1
  Nit_1             500   8.0%   5.5%      2.71   +270.9
  Station_1         500  84.5%   1.5%     -6.95   -695.3
  TAG_1             500  17.0%   9.5%      0.13    +13.2
  TAG_2             500  22.5%  15.5%      5.03   +502.8
```

Stations limp wide and bleed chips. TAGs play tight-aggressive and crush.
The numbers tell the right story.

## Layout

```
src/poker/
  cards.py       # Card, Deck, Rank, Suit
  evaluator.py   # 7-card hand strength (wraps treys)
  equity.py      # Monte Carlo equity vs random hands or specific holes
  outs.py        # outs counter, suit-specific draw probabilities
  pot_odds.py    # pot odds, required equity, EV, +EV/-EV verdict
  dashboard.py   # assembles the heads-up display
  game.py        # game state machine: Player, Action, Pot, HandState, side pots
  agents.py      # Agent abstraction + 5 bot archetypes (Nit/TAG/LAG/Maniac/Station)
  opponent_model.py  # per-opponent stats + fold-equity estimation
  preflop_ranges.py  # raw 169-class hand sets per archetype
  range_model.py     # weighted Range distributions + combo math
  archetype_ranges.py  # per-archetype open/call/3bet ranges
  range_tracker.py   # per-hand range narrowing as actions come in
  simulator.py   # multi-hand runner, JSONL dump, aggregate stats
scripts/
  demo_diamonds.py   # the classic flush-draw probability question
  walkthrough.py     # full hand street-by-street with the dashboard each street
  play.py            # interactive: you vs 5 bots, dashboard each decision
  simulate.py        # autonomous: N hands of bot vs bot, JSONL dump + stats
  narrate.py         # one hand from a chosen POV with full annotations
  compare_models.py  # A/B test: with vs without opponent modeling
tests/             # 221 tests, validates against textbook matchups
```

## Setup

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for env management.

```bash
uv sync
uv run pytest                              # 221 tests
uv run python scripts/walkthrough.py       # see the dashboard in action
uv run python scripts/play.py 50           # play 50 hands vs the bots
uv run python scripts/simulate.py 1000     # 1000-hand bot-vs-bot sim → JSONL
```

## Roadmap

- **Phase 1 — math engine** ✅
- **Phase 2 — game flow + bots + simulator** ✅
- **Phase 2.5 — opponent modeling + fold equity** ✅
  - Bots track per-opponent VPIP/PFR/fold-to-cbet/aggression
  - Postflop decisions use fold-equity-aware EV math
  - Smart archetypes (TAG/LAG/Nit) adapt; Maniac/Station deliberately don't
- **Phase 3 — ranges** ✅ — weighted hand-class distributions, per-archetype
  preflop ranges (open/call/3bet/call_3bet), `equity_vs_range()` and
  `equity_vs_ranges()` Monte Carlo, per-hand range narrowing on every
  action including postflop:
  - **Preflop:** call/raise context → call_open / threebet / call_threebet
  - **Postflop:** combo classification on the board (made_strong / made_pair /
    strong_draw / weak_draw / air); calls keep made hands + draws; raises
    keep strong made hands + per-archetype bluff weight; overpairs get full
    weight on raises (not demoted)
  - **Multiway:** when all active opponents have tracked ranges, samples
    one combo per opponent for proper multiway equity
- **Phase 4 — speed pass** — switch to a faster evaluator (phevaluator), or
  vectorize the Monte Carlo loop with numpy.
- **Phase 5 — GUI** — likely SwiftUI for the iOS path.

## Design notes

- **Engine is UI-agnostic** — `Dashboard`, `HandView`, `Action` are clean
  dataclasses ready to feed any frontend.
- **Same engine, two modes** — `Agent` is an abstract decision-maker. The
  interactive `HumanAgent` prompts via stdin; the bots decide programmatically.
  Both share the game state machine.
- **Outs are category changes**, not just kicker bumps. "Pair from high card"
  counts; "better kicker on the same pair" doesn't.
- **Equity is Monte Carlo** — bots use 800 iterations (~3% precision); demo
  scripts use 10-20k for tighter numbers.
- **Bot personalities over GTO** — for a *trainer*, learning to read player
  types beats learning to play unexploitable. Solvers are a separate problem.
- **JSONL output** captures every action, hole cards, board, and side pot,
  ready for analysis or ML training.

## Acknowledgments

Built on [treys](https://github.com/ihendley/treys) for the underlying 7-card hand evaluator.
