"""Per-hand range tracker — what we think each opponent has, narrowing as they act.

A `RangeTracker` is created at the start of each hand. As actions come in, it
narrows each opponent's range based on:
  - what action they took (call/raise/fold)
  - what archetype we believe they are (from observed stats)
  - the betting context (open vs facing-raise vs 3-bet etc.)

When a bot needs to compute equity vs an opponent, it asks the tracker for
that opponent's current range and runs equity_vs_range against it.

This is rule-based (per archetype, per action). A future Bayesian version
would compute P(action | range) and update via Bayes, but the rule-based
version is fast and approximately correct.

Reset between hands: caller creates a fresh RangeTracker each hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .archetype_ranges import (
    call_open_range, call_threebet_range, open_range, threebet_range,
)
from .cards import Card
from .opponent_model import OpponentTable, OpponentStats
from .range_model import Range, classify_combo_on_board, expand_combos


def _classify_archetype(stats: OpponentStats) -> str:
    """Bucket an opponent's observed stats into one of our 5 archetype names.

    Falls back to 'TAG' when sample is too small.
    """
    if not stats.is_reliable:
        return "TAG"   # neutral default
    vpip = stats.vpip
    if vpip < 0.13:
        return "Nit"
    if vpip < 0.24:
        return "TAG"
    if vpip < 0.40:
        return "LAG"
    if vpip < 0.65:
        return "Maniac"
    return "Station"


@dataclass
class RangeTracker:
    """Per-hand state: each opponent's currently-inferred range."""
    ranges: dict[str, Range] = field(default_factory=dict)
    archetype_by_id: dict[str, str] = field(default_factory=dict)

    def init_hand(
        self,
        opponent_ids: list[str],
        opponent_table: Optional[OpponentTable] = None,
    ) -> None:
        """Reset trackers for a new hand. Each opponent starts with a wide range
        based on their inferred archetype (from observed stats)."""
        self.ranges.clear()
        self.archetype_by_id.clear()
        # Reset the classification cache — new hand = new boards
        reset_classify_cache()
        for pid in opponent_ids:
            archetype = (
                _classify_archetype(opponent_table.get(pid))
                if opponent_table is not None else "TAG"
            )
            self.archetype_by_id[pid] = archetype
            # Start with the full open range (could be wider for free actions)
            self.ranges[pid] = open_range(archetype)

    def narrow_for_action(
        self,
        opponent_id: str,
        action_type: str,    # "fold", "call", "bet", "raise"
        street: str,         # "preflop", "flop", "turn", "river"
        is_facing_raise: bool,
        is_threebet_situation: bool = False,
        board: Optional[list[Card]] = None,
    ) -> None:
        """Update opponent's range based on observed action.

        For postflop actions, requires `board` to classify hands as
        made/draw/air on this specific board.
        """
        if opponent_id not in self.ranges:
            return  # not tracking this opponent
        archetype = self.archetype_by_id.get(opponent_id, "TAG")

        if action_type == "fold":
            # They're out — drop them from tracking
            self.ranges.pop(opponent_id, None)
            return

        if street != "preflop":
            # Postflop narrowing — needs the board
            if board is None or len(board) < 3:
                return
            self.ranges[opponent_id] = _narrow_postflop(
                self.ranges[opponent_id], board, action_type, archetype,
            )
            return

        # Preflop narrowing rules per action type
        if action_type == "call":
            if is_threebet_situation:
                self.ranges[opponent_id] = call_threebet_range(archetype)
            elif is_facing_raise:
                self.ranges[opponent_id] = call_open_range(archetype)
            else:
                # Limp (call the BB) — wider than a call-of-raise; keep open range
                # restricted to the cheaper hands (drop premium that would raise)
                from .archetype_ranges import PREMIUM
                self.ranges[opponent_id] = self.ranges[opponent_id].remove(PREMIUM)
            return

        if action_type in ("bet", "raise"):
            if is_facing_raise:
                # 3-bet (or 4-bet+) — narrow to 3-bet range
                self.ranges[opponent_id] = threebet_range(archetype)
            else:
                # Open raise — narrow to open range (was already that, no change)
                self.ranges[opponent_id] = open_range(archetype)

    def get(self, opponent_id: str) -> Optional[Range]:
        return self.ranges.get(opponent_id)


def _narrow_postflop(
    rng_range: Range,
    board: list[Card],
    action_type: str,
    archetype: str,
) -> Range:
    """Compute a new range after observing a postflop action.

    PER-COMBO weighting: each specific combo is classified on the board and
    weighted independently. This matters on flush-heavy / paired boards where
    different combos of the same class (e.g. AKo) have very different strength.

    Weight maps:
    - 'call' or 'bet': made hands (pair+) and strong draws stay at full weight;
      weak draws stay at partial weight; air drops out.
    - 'raise': strong made hands (two pair+ and overpairs) stay at full weight;
      pair gets demoted; strong draws stay at partial weight (semi-bluffs);
      air gets a small per-archetype bluff weight.
    """
    bluff_weight_by_archetype = {
        "Nit": 0.0, "TAG": 0.10, "LAG": 0.20, "Maniac": 0.40, "Station": 0.0,
    }
    bluff_w = bluff_weight_by_archetype.get(archetype, 0.10)

    if action_type in ("call", "bet"):
        weight_map = {
            "made_strong": 1.0,
            "made_pair": 1.0,
            "strong_draw": 1.0,
            "weak_draw": 0.5,
            "air": 0.0,
        }
    elif action_type == "raise":
        weight_map = {
            "made_strong": 1.0,
            "made_pair": 0.4,
            "strong_draw": 0.5,
            "weak_draw": 0.1,
            "air": bluff_w,
        }
    else:
        return rng_range

    board_set = set(board)
    # Cache: (combo_key, board_tuple) → classification. Multiple opponents on
    # the same street share the same board — this turns N narrowings × 1326
    # classifications into max ~1326 unique evaluations per street.
    cache_key_prefix = tuple(board)
    new_weights: dict = {}
    classify_cache = _classify_cache.setdefault(cache_key_prefix, {})

    for combo_key, base_w in rng_range.weights.items():
        if base_w <= 0:
            continue
        c1, c2 = tuple(combo_key)
        if c1 in board_set or c2 in board_set:
            continue   # blocked
        # Cached per-combo classification on this specific board
        cls = classify_cache.get(combo_key)
        if cls is None:
            cls = classify_combo_on_board((c1, c2), board)
            classify_cache[combo_key] = cls
        keep = weight_map.get(cls, 0.0)
        new_w = base_w * keep
        if new_w > 0.01:
            new_weights[combo_key] = new_w

    return Range(new_weights)


# Module-level classification cache. Keyed by board tuple → {combo_key: class_str}.
# Cleared between hands by reset_classify_cache().
_classify_cache: dict = {}


def reset_classify_cache() -> None:
    """Clear the classification cache (call between hands)."""
    _classify_cache.clear()
