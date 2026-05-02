"""Tests for opponent_model — running stats and fold-equity estimation."""

from poker.opponent_model import (
    MIN_RELIABLE_HANDS, OpponentStats, OpponentTable, estimate_fold_equity,
)


# ─── OpponentStats basic accessors ────────────────────────────────────────────

class TestOpponentStats:
    def test_empty_stats_use_priors(self):
        s = OpponentStats()
        # Empty stats fall back to neutral defaults
        assert s.vpip == 0.25
        assert s.pfr == 0.15
        assert s.fold_to_cbet == 0.45
        assert s.is_reliable is False

    def test_vpip_calculation(self):
        s = OpponentStats(hands_played=20, vpip_hands=5)
        assert s.vpip == 0.25

    def test_aggression_factor_no_calls(self):
        s = OpponentStats(postflop_bets_or_raises=10, postflop_calls=0)
        # Avoid div by zero — return 1.0 placeholder
        assert s.aggression_factor == 1.0

    def test_aggression_factor_calculation(self):
        s = OpponentStats(postflop_bets_or_raises=20, postflop_calls=10)
        assert s.aggression_factor == 2.0

    def test_reliable_above_threshold(self):
        s = OpponentStats(hands_played=MIN_RELIABLE_HANDS)
        assert s.is_reliable
        s2 = OpponentStats(hands_played=MIN_RELIABLE_HANDS - 1)
        assert not s2.is_reliable

    def test_str_includes_sample_warning_when_unreliable(self):
        s = OpponentStats(hands_played=5)
        assert "small sample" in str(s)


# ─── OpponentTable observation ────────────────────────────────────────────────

class TestObservation:
    def test_records_vpip_from_hand(self):
        table = OpponentTable()
        # Synthetic hand: A raises preflop, B folds, C calls
        record = {
            "players": [
                {"id": "A"}, {"id": "B"}, {"id": "C"},
            ],
            "actions": [
                {"player_id": "A", "action_type": "raise", "amount": 3, "street": "preflop"},
                {"player_id": "B", "action_type": "fold", "amount": 0, "street": "preflop"},
                {"player_id": "C", "action_type": "call", "amount": 3, "street": "preflop"},
            ],
        }
        table.observe_hand(record)
        assert table.get("A").vpip_hands == 1
        assert table.get("A").pfr_hands == 1
        assert table.get("B").vpip_hands == 0
        assert table.get("C").vpip_hands == 1
        assert table.get("C").pfr_hands == 0
        # All three count a hand played
        for pid in ("A", "B", "C"):
            assert table.get(pid).hands_played == 1

    def test_records_cbet_response(self):
        table = OpponentTable()
        record = {
            "players": [{"id": "A"}, {"id": "B"}],
            "actions": [
                # Preflop: A raises, B calls
                {"player_id": "A", "action_type": "raise", "amount": 3, "street": "preflop"},
                {"player_id": "B", "action_type": "call", "amount": 2, "street": "preflop"},
                # Flop: A cbets, B folds
                {"player_id": "A", "action_type": "bet", "amount": 5, "street": "flop"},
                {"player_id": "B", "action_type": "fold", "amount": 0, "street": "flop"},
            ],
        }
        table.observe_hand(record)
        # B faced one cbet and folded
        assert table.get("B").cbets_faced == 1
        assert table.get("B").cbets_folded_to == 1
        assert table.get("B").fold_to_cbet == 1.0

    def test_records_threebet_fold(self):
        table = OpponentTable()
        # A opens, B 3-bets, A folds
        record = {
            "players": [{"id": "A"}, {"id": "B"}],
            "actions": [
                {"player_id": "A", "action_type": "raise", "amount": 3, "street": "preflop"},
                {"player_id": "B", "action_type": "raise", "amount": 9, "street": "preflop"},
                {"player_id": "A", "action_type": "fold", "amount": 0, "street": "preflop"},
            ],
        }
        table.observe_hand(record)
        assert table.get("A").threebets_faced == 1
        assert table.get("A").threebets_folded_to == 1


# ─── Fold-equity estimation ───────────────────────────────────────────────────

class TestFoldEquityEstimation:
    def test_unreliable_stats_use_archetype_defaults(self):
        table = OpponentTable()
        # Station has cbet_fold default 0.10 — bots should not bluff stations
        fe = estimate_fold_equity("villain", table, "cbet", archetype_hint="Station")
        assert fe == 0.10
        # Nit has cbet_fold default 0.55
        fe2 = estimate_fold_equity("villain", table, "cbet", archetype_hint="Nit")
        assert fe2 == 0.55

    def test_reliable_stats_override_defaults(self):
        table = OpponentTable()
        s = table.get("villain")
        s.hands_played = MIN_RELIABLE_HANDS + 5
        s.cbets_faced = 20
        s.cbets_folded_to = 4   # 20% fold to cbet
        fe = estimate_fold_equity("villain", table, "cbet", archetype_hint="Nit")
        assert abs(fe - 0.20) < 0.01   # observed beats archetype hint

    def test_no_archetype_hint_uses_population(self):
        table = OpponentTable()
        fe = estimate_fold_equity("villain", table, "cbet")
        assert fe == 0.45  # population default
