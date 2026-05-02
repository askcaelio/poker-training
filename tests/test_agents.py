"""Tests for agents — bot decisions and the agent abstraction."""

import random

import pytest

from poker.cards import Deck, parse_cards
from poker.game import (
    Action, ActionType, Player, apply_action, award_pots, start_hand,
)
from poker.agents import (
    AlwaysFoldAgent, BotAgent, BotConfig, RandomAgent,
    NIT_RANGE, TAG_RANGE, LAG_RANGE, MANIAC_RANGE, STATION_RANGE,
    hand_class, make_lag, make_maniac, make_nit, make_station, make_tag,
)


def deck(seed: int = 42) -> Deck:
    return Deck(rng=random.Random(seed))


# ─── Hand classification ──────────────────────────────────────────────────────

class TestHandClass:
    def test_pocket_pair(self):
        h = parse_cards("As Ah")
        assert hand_class(h[0], h[1]) == "AA"

    def test_suited(self):
        h = parse_cards("As Ks")
        assert hand_class(h[0], h[1]) == "AKs"

    def test_offsuit(self):
        h = parse_cards("As Kh")
        assert hand_class(h[0], h[1]) == "AKo"

    def test_low_card_first(self):
        h = parse_cards("2c Ks")
        assert hand_class(h[0], h[1]) == "K2o"

    def test_suited_low(self):
        h = parse_cards("5d 2d")
        assert hand_class(h[0], h[1]) == "52s"


# ─── Range definitions ────────────────────────────────────────────────────────

class TestRanges:
    def test_nit_includes_premium(self):
        for h in ["AA", "KK", "QQ", "AKs"]:
            assert h in NIT_RANGE
        # Excludes weak hands
        for h in ["72o", "83s", "T2o"]:
            assert h not in NIT_RANGE

    def test_tag_superset_of_nit(self):
        assert NIT_RANGE.issubset(TAG_RANGE)

    def test_lag_superset_of_tag(self):
        assert TAG_RANGE.issubset(LAG_RANGE)

    def test_maniac_superset_of_lag(self):
        assert LAG_RANGE.issubset(MANIAC_RANGE)

    def test_station_widest(self):
        assert MANIAC_RANGE.issubset(STATION_RANGE)


# ─── Agent decision sanity ────────────────────────────────────────────────────

class TestAgentDecisions:
    def test_nit_folds_72o_preflop(self):
        # Set up state where actor has 72o
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        # Manually set hero's hole cards to 72o for this test
        p1.hole_cards = tuple(parse_cards("7c 2d"))
        view = s.view_for("hero")
        nit = make_nit(rng=random.Random(0))
        action = nit.decide(view)
        assert action.type == ActionType.FOLD

    def test_nit_plays_AA_preflop(self):
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        p1.hole_cards = tuple(parse_cards("As Ah"))
        view = s.view_for("hero")
        nit = make_nit(rng=random.Random(0))
        action = nit.decide(view)
        # Either calls or raises — definitely doesn't fold AA
        assert action.type != ActionType.FOLD

    def test_station_plays_garbage_preflop(self):
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        p1.hole_cards = tuple(parse_cards("9c 4h"))   # 94o
        view = s.view_for("hero")
        station = make_station(rng=random.Random(0))
        action = station.decide(view)
        # Station should play 94o (it's in MANIAC ⊂ STATION); they call
        assert action.type in (ActionType.CALL, ActionType.RAISE)


# ─── Always-fold agent ────────────────────────────────────────────────────────

class TestAlwaysFold:
    def test_fold_to_bet(self):
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        view = s.view_for(s.actor().id)
        agent = AlwaysFoldAgent()
        assert agent.decide(view).type == ActionType.FOLD

    def test_check_when_free(self):
        # Need a state where to_call == 0
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        # SB calls, BB checks → flop, BB acts first, to_call=0
        apply_action(s, Action(ActionType.CALL, 0.5))
        apply_action(s, Action(ActionType.CHECK))
        view = s.view_for(s.actor().id)
        agent = AlwaysFoldAgent()
        assert agent.decide(view).type == ActionType.CHECK


# ─── Random agent (smoke) ─────────────────────────────────────────────────────

class TestRandomAgent:
    def test_returns_legal_action(self):
        p1 = Player(id="hero", stack=100.0)
        p2 = Player(id="vil", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        view = s.view_for(s.actor().id)
        agent = RandomAgent(rng=random.Random(7))
        action = agent.decide(view)
        # Should be one of fold/call/raise (since facing the BB)
        assert action.type in {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}


# ─── Full hand: bots play to completion ───────────────────────────────────────

class TestFullHandPlayThrough:
    def test_two_nits_play_full_hand(self):
        # Two nits — should have a deterministic-ish hand (with seed)
        p1 = Player(id="A", stack=100.0)
        p2 = Player(id="B", stack=100.0)
        s = start_hand([p1, p2], 0, deck(42), (0.5, 1.0))
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        agents = {"A": make_nit("A", rng=rng_a), "B": make_nit("B", rng=rng_b)}

        max_actions = 60
        n = 0
        while not s.is_complete and n < max_actions:
            actor = s.actor()
            view = s.view_for(actor.id)
            action = agents[actor.id].decide(view)
            apply_action(s, action)
            n += 1
        assert s.is_complete
        awards = award_pots(s)
        # Pot was awarded
        assert sum(a.amount for a in awards) > 0
        # Stacks plus pot equals starting total
        total_chips = sum(p.stack for p in [p1, p2])
        assert total_chips == 200.0  # original 100+100; pot already added back to winner

    def test_six_bots_play_full_hand(self):
        # Mixed table: 6 bots play one hand to completion
        players = [Player(id=f"p{i}", stack=100.0) for i in range(6)]
        s = start_hand(players, 0, deck(123), (0.5, 1.0))
        agents = {
            "p0": make_nit("p0", rng=random.Random(10)),
            "p1": make_tag("p1", rng=random.Random(11)),
            "p2": make_lag("p2", rng=random.Random(12)),
            "p3": make_tag("p3", rng=random.Random(13)),
            "p4": make_maniac("p4", rng=random.Random(14)),
            "p5": make_station("p5", rng=random.Random(15)),
        }
        max_actions = 100
        n = 0
        while not s.is_complete and n < max_actions:
            actor = s.actor()
            view = s.view_for(actor.id)
            action = agents[actor.id].decide(view)
            apply_action(s, action)
            n += 1
        assert s.is_complete
        awards = award_pots(s)
        # Chip conservation: total = 600
        total_chips = sum(p.stack for p in players)
        assert total_chips == 600.0
