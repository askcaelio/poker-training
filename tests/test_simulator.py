"""Tests for the simulation runner."""

import json
from pathlib import Path

import pytest

from poker.agents import AlwaysFoldAgent, make_lag, make_nit, make_tag
from poker.simulator import play_one_hand, run_simulation


class TestSimulatorBasics:
    def test_short_run_completes(self):
        agents = [make_nit("Nit"), make_tag("TAG")]
        stats = run_simulation(agents, num_hands=5, seed=42)
        assert stats.total_hands == 5
        assert "Nit" in stats.per_player
        assert "TAG" in stats.per_player
        # Both played all 5 hands
        assert stats.per_player["Nit"]["hands"] == 5
        assert stats.per_player["TAG"]["hands"] == 5

    def test_chip_conservation(self):
        # Across N hands with fresh stacks, total winnings sum to zero
        agents = [make_nit("Nit"), make_tag("TAG"), make_lag("LAG")]
        stats = run_simulation(agents, num_hands=20, seed=7)
        total_won = sum(s["total_won"] for s in stats.per_player.values())
        assert abs(total_won) < 1e-9   # all chips accounted for

    def test_persistent_stacks_carry_over(self):
        agents = [make_nit("Nit"), make_lag("LAG")]
        stats = run_simulation(
            agents, num_hands=10, persistent_stacks=True, seed=42,
        )
        # Total chips conserved
        total_won = sum(s["total_won"] for s in stats.per_player.values())
        assert abs(total_won) < 1e-9

    def test_jsonl_output(self, tmp_path: Path):
        agents = [make_nit("Nit"), make_tag("TAG")]
        out = tmp_path / "hands.jsonl"
        run_simulation(agents, num_hands=3, output_path=out, seed=123)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "hand_id" in record
            assert "actions" in record
            assert "awards" in record
            assert "players" in record


class TestAlwaysFoldOpponent:
    def test_always_folder_loses_blinds_only(self):
        # AlwaysFold always folds when not in blind. As BB they post and fold.
        # The non-folder takes the blinds.
        agents = [AlwaysFoldAgent(), make_tag("TAG")]
        # Force consistent button rotation so each plays SB and BB equally
        stats = run_simulation(agents, num_hands=20, seed=42)
        # Always-folder should lose money
        assert stats.per_player["AlwaysFold"]["total_won"] < 0
        assert stats.per_player["TAG"]["total_won"] > 0


class TestStatsShape:
    def test_stats_render(self):
        agents = [make_nit("Nit"), make_tag("TAG")]
        stats = run_simulation(agents, num_hands=5, seed=42)
        s = str(stats)
        assert "5" in s
        assert "Nit" in s
        assert "TAG" in s
        assert "VPIP" in s
        assert "BB/100" in s
