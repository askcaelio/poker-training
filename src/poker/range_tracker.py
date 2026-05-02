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
from .opponent_model import OpponentTable, OpponentStats
from .range_model import Range


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
    ) -> None:
        """Update opponent's range based on observed action."""
        if opponent_id not in self.ranges:
            return  # not tracking this opponent
        archetype = self.archetype_by_id.get(opponent_id, "TAG")

        if action_type == "fold":
            # They're out — drop them from tracking
            self.ranges.pop(opponent_id, None)
            return

        if street != "preflop":
            # Postflop: simple narrowing — calls keep "made hand or draw" hands;
            # raises keep "strong made hands or bluffs". We're not analyzing
            # board texture yet, so use a coarse approximation: don't narrow
            # postflop in v1. Future: detect pairs/draws/air on the board.
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
