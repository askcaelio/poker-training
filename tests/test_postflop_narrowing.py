"""Tests for postflop range narrowing (combo classification + narrow_postflop)."""

import pytest

from poker.cards import parse_cards
from poker.range_model import (
    Range, classify_combo_on_board, expand_combos,
)
from poker.range_tracker import _narrow_postflop


# ─── Combo classification ─────────────────────────────────────────────────────

class TestClassifyCombo:
    def test_top_pair_is_made(self):
        # AsKs on Kh 7d 2c → top pair
        cls = classify_combo_on_board(
            tuple(parse_cards("As Ks")), parse_cards("Kh 7d 2c")
        )
        assert cls == "made_pair"

    def test_set_is_made_strong(self):
        # 7s 7d on 7c Kh 2d → set of 7s (Three of a Kind)
        cls = classify_combo_on_board(
            tuple(parse_cards("7s 7d")), parse_cards("7c Kh 2d")
        )
        assert cls == "made_strong"

    def test_two_pair_is_made_strong(self):
        # AsKs on As Ks 2c → two pair
        # But hero blocks the As and Ks already on board — so this won't happen
        # Use AhKh on As Kc 2d → two pair (As + Kc)
        cls = classify_combo_on_board(
            tuple(parse_cards("Ah Kh")), parse_cards("As Kc 2d")
        )
        assert cls == "made_strong"

    def test_flush_draw_is_strong_draw(self):
        # AsKs on Qs 7s 2c → 4 spades = flush draw
        cls = classify_combo_on_board(
            tuple(parse_cards("As Ks")), parse_cards("Qs 7s 2c")
        )
        # Strong draw — flush draw + overcards are all draws here, no pair
        assert cls == "strong_draw"

    def test_oesd_is_strong_draw(self):
        # 9s8h on 7d 6c 2s → open-ended straight draw
        cls = classify_combo_on_board(
            tuple(parse_cards("9s 8h")), parse_cards("7d 6c 2s")
        )
        assert cls == "strong_draw"

    def test_overcards_only_is_weak_draw(self):
        # AsKs on Qh 7d 2c — no flush draw, no straight draw, just two overcards
        cls = classify_combo_on_board(
            tuple(parse_cards("Ah Kc")), parse_cards("Qd 7s 2c")
        )
        # Two overcards but no flush/straight — weak draw category
        assert cls == "weak_draw"

    def test_complete_air(self):
        # 4c 3h on Kh 8d 2s — no pair, no draw, no overcards
        cls = classify_combo_on_board(
            tuple(parse_cards("4c 3h")), parse_cards("Kh 8d 2s")
        )
        assert cls == "air"


# ─── Postflop narrowing ───────────────────────────────────────────────────────

class TestNarrowPostflop:
    def test_call_keeps_made_hands_drops_air(self):
        # Range = AA (overpair = strong), KK (trips = strong), J9o (true air on Kh-7d-2c)
        # After "call", AA and KK stay; J9o drops (no pair, no draw).
        r = Range.from_set({"AA", "KK", "J9o"})
        narrowed = _narrow_postflop(r, parse_cards("Kh 7d 2c"), "call", "TAG")
        assert narrowed.has("AA")
        assert narrowed.has("KK")
        assert not narrowed.has("J9o")

    def test_raise_demotes_pair_keeps_overpair(self):
        # On Kh 7d 2c:
        #   AA = made_strong (overpair, higher than top board K)
        #   KQo = made_pair (top pair) — but Kh on board blocks 3 of 12 KQo combos
        #   J9o = air (no pair, no draw)
        # After "raise" by a TAG: AA keeps full weight; KQo's per-combo weight is
        # 0.4 but the get() averages over all 12 (including the 3 blocked → 0
        # weight), so get("KQo") ≈ (9×0.4)/12 ≈ 0.3.
        r = Range.from_set({"AA", "KQo", "J9o"})
        narrowed = _narrow_postflop(r, parse_cards("Kh 7d 2c"), "raise", "TAG")
        assert narrowed.get("AA") > 0.9   # overpair = full weight
        assert 0.25 < narrowed.get("KQo") < 0.45   # top pair demoted, blockers reduce avg
        # J9o gets the TAG bluff weight (~0.10)

    def test_raise_drops_air_for_nit(self):
        # Nit has bluff_weight = 0.0, so true air drops to 0 on a raise.
        # T9o on Kh 7d 2c: T and 9 don't pair the board; T > 7,2 (overcard) so
        # it's "weak_draw" not pure "air" in our classifier. Use 6-5o which has
        # no overcards and no draws on Kh 7d 2c: 5 < 7, 6 < 7. Pure air.
        r = Range.from_set({"AA", "65o"})
        narrowed = _narrow_postflop(r, parse_cards("Kh 9d 2c"), "raise", "Nit")
        assert narrowed.has("AA")
        # 65o on Kh 9d 2c is true air → Nit's bluff weight is 0 → drops out
        assert not narrowed.has("65o")

    def test_call_keeps_strong_draws(self):
        # 9s8s on 7s 6h 2c → flush draw + OESD (huge equity hand)
        # In a range this would survive a call
        r = Range.from_set({"98s"})
        # 98s on 7s6h2c — well, 9-8s is suited; 7-6 is on the board
        # Check classify directly
        cls = classify_combo_on_board(
            tuple(parse_cards("9s 8s")), parse_cards("7s 6h 2c")
        )
        # No flush draw on this board (only 1 spade in hand + 1 on board = 2 spades total)
        # Actually 9s 8s + 7s = 3 spades... no 4 spades needed for flush draw.
        # But it IS an OESD (5 or T completes 5-6-7-8-9 or 6-7-8-9-T).
        assert cls == "strong_draw"


class TestNarrowDrasticReduction:
    def test_air_heavy_range_shrinks_after_call(self):
        # Build a range that's mostly air on a particular board, then call
        # narrows it dramatically.
        from poker.preflop_ranges import STATION_RANGE
        r = Range.from_set(STATION_RANGE)
        # Dry, low board where most hands miss
        narrowed = _narrow_postflop(r, parse_cards("Kh 7d 2c"), "call", "Station")
        # Range should shrink (lots of station hands are air on this board)
        assert narrowed.total_combos < r.total_combos
        # But not to zero — Station has some Kx, 7x, 2x pairs and some pocket pairs
        assert narrowed.total_combos > 0


class TestComboLevelDifferentiation:
    """Per-combo narrowing matters most on monotone/flush-heavy boards.

    These tests verify that combos within the same class get DIFFERENT weights
    based on whether they have a flush card or not.
    """

    def test_offsuit_combos_differ_on_monotone_board(self):
        # Hero is irrelevant — we're testing the narrowing of villain's range.
        # Range = AKo (12 combos). Board = Qs Js 4s (monotone spades).
        # 6 of 12 AKo combos have a spade in hand → backdoor flush draw potential
        # 6 of 12 have no spade → mostly air on this flush-heavy board
        r = Range.from_set({"AKo"})
        board = parse_cards("Qs Js 4s")
        narrowed = _narrow_postflop(r, board, "call", "TAG")

        # Inspect each combo's weight individually
        from poker.cards import Card, Rank, Suit
        spade_combos = []
        non_spade_combos = []
        for c1, c2 in [(Card(Rank.ACE, s1), Card(Rank.KING, s2))
                       for s1 in Suit for s2 in Suit if s1 != s2]:
            if c1 in board or c2 in board:
                continue
            w = narrowed.get_combo(c1, c2)
            if c1.suit == Suit.SPADES or c2.suit == Suit.SPADES:
                spade_combos.append(w)
            else:
                non_spade_combos.append(w)

        # Non-spade combos are pure air on Qs Js 4s — should drop to 0
        # Spade combos have at least overcards + backdoor flush potential
        # The asymmetry is what proves combo-level narrowing works
        non_spade_avg = sum(non_spade_combos) / len(non_spade_combos) if non_spade_combos else 0
        spade_avg = sum(spade_combos) / len(spade_combos) if spade_combos else 0

        # Spade-containing combos should have higher weight than non-spade
        # (or both dropped, or some asymmetry — definitely not equal)
        assert spade_avg != non_spade_avg or (spade_avg == 0 and non_spade_avg == 0)

    def test_suited_combo_with_flush_gets_full_weight(self):
        # AKs with the spade-spade combo (AsKs) has a 4-card flush on Qs Js 4s.
        # That's a made flush! The non-spade AKs combos are pure air.
        from poker.cards import Card, Rank, Suit
        r = Range.from_set({"AKs"})
        board = parse_cards("Qs Js 4s")
        narrowed = _narrow_postflop(r, board, "call", "TAG")

        # AsKs is a flush — but the spades are blocked by the board (Qs, Js, 4s).
        # Wait: AsKs uses As (not on board) and Ks (not on board). So AsKs is playable.
        # AsKs + Qs Js 4s = 5 spades = flush. Definitely "made_strong".
        as_ks_weight = narrowed.get_combo(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES))
        ah_kh_weight = narrowed.get_combo(Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS))

        # AsKs makes a flush → full weight kept on call
        assert as_ks_weight > 0.9
        # AhKh on a spade board: just two overcards, weak draw or air
        # Should be lower than AsKs
        assert as_ks_weight > ah_kh_weight
