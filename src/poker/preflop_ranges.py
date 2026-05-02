"""Raw preflop range definitions (sets of 169-class hand strings).

Lives in its own module to break circular imports between agents.py and
archetype_ranges.py — both need these constants but neither should depend
on the other.
"""

from __future__ import annotations


# Hands ordered roughly by strength. Strongest archetype-included only.
NIT_RANGE: set[str] = {
    "AA", "KK", "QQ", "JJ", "TT", "99",
    "AKs", "AKo", "AQs", "AQo", "AJs",
    "KQs",
}  # ~6% of hands

TAG_RANGE: set[str] = NIT_RANGE | {
    "88", "77", "66", "55", "44", "33", "22",
    "AJo", "ATs", "ATo", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQo", "KJs", "KJo", "KTs", "K9s",
    "QJs", "QJo", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s", "97s", "87s", "86s", "76s", "75s", "65s", "54s",
}  # ~18-20%

LAG_RANGE: set[str] = TAG_RANGE | {
    "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "Q8s", "Q7s", "Q6s", "Q5s",
    "J8s", "J7s",
    "T7s", "T6s",
    "96s", "85s", "74s", "64s", "53s", "43s",
    "ATo", "K9o", "Q9o", "J9o", "T9o", "98o", "87o", "76o",
}  # ~30-32%

MANIAC_RANGE: set[str] = LAG_RANGE | {
    "Q4s", "Q3s", "Q2s", "J6s", "J5s", "J4s", "J3s", "J2s",
    "T5s", "T4s", "T3s", "T2s",
    "95s", "94s", "93s", "92s", "84s", "83s", "82s", "73s", "72s",
    "63s", "62s", "52s", "42s", "32s",
    "K8o", "K7o", "K6o", "Q8o", "Q7o", "J8o", "T8o", "97o", "86o", "75o", "65o", "54o",
}  # ~50%

STATION_RANGE: set[str] = MANIAC_RANGE | {
    "K5o", "K4o", "K3o", "K2o", "Q6o", "Q5o", "Q4o", "Q3o", "Q2o",
    "J7o", "J6o", "J5o", "J4o", "J3o", "J2o",
    "T7o", "T6o", "T5o", "T4o", "T3o", "T2o",
    "96o", "95o", "94o", "93o", "92o",
    "85o", "84o", "83o", "82o", "74o", "73o", "72o",
    "64o", "63o", "62o", "53o", "52o", "42o", "32o",
}  # ~85%
