"""Opponent modeling — running statistics that bots use to estimate fold equity.

The premise: every observable action tells you something about a player. After
~30 hands, you know who's a station (fold-to-cbet ≈ 5%), who's a nit (fold to
3-bet ≈ 90%), who's a maniac (cbets 90% with anything). Smart bots use this
to size bets and pick spots.

Stats tracked per opponent:
  - VPIP:           % of hands the player voluntarily put money in preflop
  - PFR:            % of hands where they raised preflop
  - fold-to-cbet:   when facing a flop cbet, how often they folded
  - fold-to-3bet:   when facing a 3-bet preflop, how often they folded
  - aggression:     (bets + raises) / calls — postflop aggression factor

`estimate_fold_equity()` translates these into a fold-probability for a
specific betting situation. When a bot doesn't have enough data (< 15 hands),
it falls back to a default by archetype guess.

This is a "shared whiteboard" model: every player at the table sees the same
stats (everyone observes the same actions). Each bot reads from the same
OpponentTable. That's deliberate — opponent stats are public information,
not privileged knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Below this many hands of data, we don't trust the per-opponent stats.
# Use a default fold-equity assumption instead.
MIN_RELIABLE_HANDS = 15


@dataclass
class OpponentStats:
    """Running stats for a single opponent."""
    hands_played: int = 0
    vpip_hands: int = 0          # hands they voluntarily put money in preflop
    pfr_hands: int = 0           # hands they raised preflop
    cbets_faced: int = 0         # times they faced a flop cbet
    cbets_folded_to: int = 0     # times they folded to a flop cbet
    threebets_faced: int = 0     # times they faced a 3-bet preflop
    threebets_folded_to: int = 0
    postflop_bets_or_raises: int = 0
    postflop_calls: int = 0

    @property
    def is_reliable(self) -> bool:
        return self.hands_played >= MIN_RELIABLE_HANDS

    @property
    def vpip(self) -> float:
        if self.hands_played == 0:
            return 0.25  # neutral prior
        return self.vpip_hands / self.hands_played

    @property
    def pfr(self) -> float:
        if self.hands_played == 0:
            return 0.15
        return self.pfr_hands / self.hands_played

    @property
    def fold_to_cbet(self) -> float:
        if self.cbets_faced == 0:
            return 0.45  # population default
        return self.cbets_folded_to / self.cbets_faced

    @property
    def fold_to_3bet(self) -> float:
        if self.threebets_faced == 0:
            return 0.55  # population default
        return self.threebets_folded_to / self.threebets_faced

    @property
    def aggression_factor(self) -> float:
        """Bets + raises per call (postflop). >2 = aggressive, <1 = passive."""
        if self.postflop_calls == 0:
            return 1.0
        return self.postflop_bets_or_raises / self.postflop_calls

    def __str__(self) -> str:
        rel = "" if self.is_reliable else " (small sample)"
        return (
            f"{self.hands_played}h  "
            f"VPIP {self.vpip * 100:.0f}%  PFR {self.pfr * 100:.0f}%  "
            f"FvCB {self.fold_to_cbet * 100:.0f}%  "
            f"AF {self.aggression_factor:.1f}{rel}"
        )


class OpponentTable:
    """Collection of OpponentStats, one per player id."""

    def __init__(self) -> None:
        self._players: dict[str, OpponentStats] = {}

    def get(self, player_id: str) -> OpponentStats:
        if player_id not in self._players:
            self._players[player_id] = OpponentStats()
        return self._players[player_id]

    def all_players(self) -> dict[str, OpponentStats]:
        return dict(self._players)

    def observe_hand(self, hand_record: dict) -> None:
        """Update stats from a serialized hand record (as produced by the simulator).

        Expects the dict shape `play_one_hand` returns: keys `players`,
        `actions` (each with player_id/action_type/street).
        """
        # Per-hand flags (avoid double-counting multi-action hands)
        vpip_this_hand: set[str] = set()
        pfr_this_hand: set[str] = set()
        # Track who cbet the flop (the preflop aggressor) and who faced it
        preflop_aggressor: Optional[str] = None
        preflop_aggressor_streets_seen = False
        first_flop_bettor: Optional[str] = None
        last_preflop_raiser: Optional[str] = None
        preflop_raises: int = 0

        # Touch every player in the hand so they get a hands_played increment
        for snap in hand_record["players"]:
            self.get(snap["id"]).hands_played += 1

        actions = hand_record["actions"]

        # First pass: preflop stats + identify aggressors
        for rec in actions:
            if rec["street"] != "preflop":
                continue
            pid = rec["player_id"]
            atype = rec["action_type"]
            if atype in ("call", "bet", "raise"):
                vpip_this_hand.add(pid)
            if atype in ("bet", "raise"):
                pfr_this_hand.add(pid)
                preflop_raises += 1
                if last_preflop_raiser is not None:
                    # This is a 3-bet (or higher). The previous raiser is the
                    # one being 3-bet against.
                    self.get(last_preflop_raiser).threebets_faced += 1
                last_preflop_raiser = pid

        # Detect 3-bet folds: if last_preflop_raiser was 3-bet and they folded
        # ...we track that by looking at their next preflop action after the 3-bet
        # Simpler: count fold-to-3bet by checking each player's response to the 3rd raise+
        if preflop_raises >= 2:
            # Track players who folded after facing the 3-bet
            seen_threebet = False
            three_bettor: Optional[str] = None
            for rec in actions:
                if rec["street"] != "preflop":
                    continue
                pid = rec["player_id"]
                atype = rec["action_type"]
                if atype in ("bet", "raise"):
                    if seen_threebet:
                        pass  # 4-bet — out of scope for 3-bet fold tracking
                    else:
                        # First raise was the open; second raise is the 3-bet
                        if three_bettor is None and pid != _first_raiser_id(actions):
                            three_bettor = pid
                            seen_threebet = True
                elif atype == "fold" and seen_threebet:
                    # This player folded after seeing a 3-bet
                    self.get(pid).threebets_folded_to += 1

        # Identify preflop aggressor (last player to bet/raise preflop) for cbet logic
        preflop_aggressor = last_preflop_raiser

        # Apply VPIP / PFR
        for pid in vpip_this_hand:
            self.get(pid).vpip_hands += 1
        for pid in pfr_this_hand:
            self.get(pid).pfr_hands += 1

        # Cbet tracking: first bet on the flop by the preflop aggressor counts as a cbet
        if preflop_aggressor is not None:
            saw_flop_action = False
            cbet_made = False
            cbet_responses_collected: set[str] = set()
            for rec in actions:
                if rec["street"] != "flop":
                    continue
                saw_flop_action = True
                pid = rec["player_id"]
                atype = rec["action_type"]
                if not cbet_made:
                    if pid == preflop_aggressor and atype in ("bet", "raise"):
                        cbet_made = True
                    elif atype in ("bet", "raise"):
                        # Someone else bet first — not a cbet from the PFR
                        break
                else:
                    # Already cbet — track responses
                    if pid in cbet_responses_collected:
                        continue
                    cbet_responses_collected.add(pid)
                    self.get(pid).cbets_faced += 1
                    if atype == "fold":
                        self.get(pid).cbets_folded_to += 1

        # Aggression factor: count postflop bets/raises and calls per player
        for rec in actions:
            if rec["street"] == "preflop":
                continue
            pid = rec["player_id"]
            atype = rec["action_type"]
            stats = self.get(pid)
            if atype in ("bet", "raise"):
                stats.postflop_bets_or_raises += 1
            elif atype == "call":
                stats.postflop_calls += 1


def _first_raiser_id(actions: list[dict]) -> Optional[str]:
    """Helper: id of the first preflop raiser (None if no one raised)."""
    for rec in actions:
        if rec["street"] == "preflop" and rec["action_type"] in ("bet", "raise"):
            return rec["player_id"]
    return None


def estimate_fold_equity(
    opponent_id: str,
    table: OpponentTable,
    context: str,
    archetype_hint: Optional[str] = None,
) -> float:
    """Estimate P(fold) for a specific betting situation.

    Args:
        opponent_id: who we're betting at
        table: opponent stats
        context: one of "cbet", "preflop_open", "preflop_3bet", "barrel" (turn/river)
        archetype_hint: optional fallback when stats are unreliable

    Returns:
        Fold probability in [0, 1].
    """
    stats = table.get(opponent_id)

    if stats.is_reliable:
        if context == "cbet":
            return stats.fold_to_cbet
        if context == "preflop_open":
            return 1.0 - stats.vpip
        if context == "preflop_3bet":
            return stats.fold_to_3bet
        if context == "barrel":
            # Rough proxy: passive players fold more on later streets
            return min(0.7, stats.fold_to_cbet * 1.1)

    # Insufficient data — use archetype hint or population defaults
    defaults = {
        "Nit":     {"cbet": 0.55, "preflop_open": 0.85, "preflop_3bet": 0.85, "barrel": 0.60},
        "TAG":     {"cbet": 0.50, "preflop_open": 0.65, "preflop_3bet": 0.65, "barrel": 0.50},
        "LAG":     {"cbet": 0.40, "preflop_open": 0.50, "preflop_3bet": 0.50, "barrel": 0.40},
        "Maniac":  {"cbet": 0.25, "preflop_open": 0.30, "preflop_3bet": 0.40, "barrel": 0.20},
        "Station": {"cbet": 0.10, "preflop_open": 0.15, "preflop_3bet": 0.30, "barrel": 0.10},
    }
    if archetype_hint and archetype_hint in defaults:
        return defaults[archetype_hint].get(context, 0.40)
    # Generic population default
    return {"cbet": 0.45, "preflop_open": 0.55, "preflop_3bet": 0.55, "barrel": 0.40}.get(context, 0.40)
