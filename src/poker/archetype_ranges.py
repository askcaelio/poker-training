"""Pre-defined preflop ranges per archetype × action.

For each archetype (Nit, TAG, LAG, Maniac, Station), we define four ranges:
  - OPEN:        hands they raise when they're first to enter the pot
  - CALL_OPEN:   hands they call after someone else opens
  - THREEBET:    hands they 3-bet over an open
  - CALL_THREEBET: hands they call vs a 3-bet (4-betting only with the very top)

These are textbook approximations, not GTO. They mirror the BotConfig logic:
the same hands the bot would PLAY are roughly the same hands it RANGES on
its opponents. Bots assume opponents play their archetype.

Range narrowing: when villain takes an action, we restrict their range to the
sub-range that matches that action.
"""

from __future__ import annotations

from .preflop_ranges import (
    LAG_RANGE, MANIAC_RANGE, NIT_RANGE, STATION_RANGE, TAG_RANGE,
)
from .range_model import Range, all_hand_classes


# ─── Helper: build a "top-N%" range using NIT⊂TAG⊂LAG⊂MANIAC⊂STATION ────────

# Premium "nuts" hands (used as the default 3-bet range)
PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
# Strong (used for 4-bet / call-3bet range for tight players)
STRONG = PREMIUM | {"TT", "AQs", "AQo"}


# ─── Per-archetype ranges as Range objects (weight 1.0) ──────────────────────

NIT_OPEN = Range.from_set(NIT_RANGE)
TAG_OPEN = Range.from_set(TAG_RANGE)
LAG_OPEN = Range.from_set(LAG_RANGE)
MANIAC_OPEN = Range.from_set(MANIAC_RANGE)
STATION_OPEN = Range.from_set(STATION_RANGE)

# Calling-an-open ranges: typically slightly tighter than opening (you call
# with hands not strong enough to 3-bet, drop the trash that you'd fold to
# pressure). For simplicity, use the open range minus the 3-bet range.
NIT_CALL_OPEN = Range.from_set(NIT_RANGE - PREMIUM)
TAG_CALL_OPEN = Range.from_set(TAG_RANGE - PREMIUM)
LAG_CALL_OPEN = Range.from_set(LAG_RANGE - PREMIUM)
MANIAC_CALL_OPEN = Range.from_set(MANIAC_RANGE)         # maniacs don't fold much; call wide
STATION_CALL_OPEN = Range.from_set(STATION_RANGE)       # stations call everything they opened with

# 3-bet ranges: typically only the premium portion of opening range
# (plus some bluffs for the wider archetypes)
NIT_THREEBET = Range.from_set({"AA", "KK", "QQ", "AKs"})
TAG_THREEBET = Range.from_set(PREMIUM)
LAG_THREEBET = Range.from_set(PREMIUM | {"AQs", "TT", "A5s", "K9s"})  # add bluffs
MANIAC_THREEBET = Range.from_set(PREMIUM | {"AQs", "AQo", "TT", "99", "JJ", "AJs", "KQs"})
STATION_THREEBET = Range.from_set({"AA", "KK", "QQ"})   # only nut hands

# Range that calls a 3-bet (vs your own open getting 3-bet)
NIT_CALL_THREEBET = Range.from_set({"JJ", "AKo"})
TAG_CALL_THREEBET = Range.from_set({"JJ", "TT", "AKo", "AQs"})
LAG_CALL_THREEBET = Range.from_set({"TT", "99", "88", "AQo", "AKo", "AJs", "KQs"})
MANIAC_CALL_THREEBET = Range.from_set(MANIAC_RANGE - PREMIUM)  # calls wide
STATION_CALL_THREEBET = Range.from_set(STATION_RANGE - {"AA", "KK"})  # calls everything they had


# ─── Archetype lookup ────────────────────────────────────────────────────────

def open_range(archetype: str) -> Range:
    return {
        "Nit": NIT_OPEN, "TAG": TAG_OPEN, "LAG": LAG_OPEN,
        "Maniac": MANIAC_OPEN, "Station": STATION_OPEN,
    }.get(archetype, TAG_OPEN)


def call_open_range(archetype: str) -> Range:
    return {
        "Nit": NIT_CALL_OPEN, "TAG": TAG_CALL_OPEN, "LAG": LAG_CALL_OPEN,
        "Maniac": MANIAC_CALL_OPEN, "Station": STATION_CALL_OPEN,
    }.get(archetype, TAG_CALL_OPEN)


def threebet_range(archetype: str) -> Range:
    return {
        "Nit": NIT_THREEBET, "TAG": TAG_THREEBET, "LAG": LAG_THREEBET,
        "Maniac": MANIAC_THREEBET, "Station": STATION_THREEBET,
    }.get(archetype, TAG_THREEBET)


def call_threebet_range(archetype: str) -> Range:
    return {
        "Nit": NIT_CALL_THREEBET, "TAG": TAG_CALL_THREEBET, "LAG": LAG_CALL_THREEBET,
        "Maniac": MANIAC_CALL_THREEBET, "Station": STATION_CALL_THREEBET,
    }.get(archetype, TAG_CALL_THREEBET)
