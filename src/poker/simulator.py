"""Simulation runner — autonomous bot vs bot at scale.

`run_simulation()` plays N hands at a table of agents, dumping every action
and showdown to a JSONL file. Stats (VPIP, PFR, BB/100, win rate) are computed
from the dump.

Stack handling: each player starts each hand with a fresh stack (default 100bb)
unless `persistent_stacks=True`, in which case stacks carry over and busted
players are removed (or refreshed if `auto_rebuy=True`).

The button rotates each hand (cash-game convention).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, TextIO

from .cards import Card, Deck
from .agents import Agent, BotAgent
from .game import (
    Action, ActionRecord, HandState, Player, PotAward,
    apply_action, award_pots, start_hand,
)
from .opponent_model import OpponentTable


@dataclass
class HandResult:
    """One hand's worth of result data — what gets serialized to JSONL."""
    hand_id: int
    button_player_id: str
    blinds: tuple[float, float]
    players: list[dict]                    # id, position, hole, starting_stack, ending_stack
    board: list[str]
    actions: list[dict]                    # serialized ActionRecord list
    awards: list[dict]                     # PotAwards
    final_pot: float


@dataclass
class SimStats:
    """Aggregate stats across many hands."""
    total_hands: int = 0
    per_player: dict[str, dict] = field(default_factory=dict)
    opponent_table: Optional["OpponentTable"] = None

    def __str__(self) -> str:
        lines = [f"Sim: {self.total_hands:,} hands\n"]
        header = f"  {'player':<14} {'hands':>6} {'VPIP':>6} {'PFR':>6} {'FvCB':>6} {'AF':>5} {'won/hand':>10} {'BB/100':>8}"
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))
        for pid, s in sorted(self.per_player.items()):
            # Pull observed-by-others FvCB/AF from the shared opponent table if we have it
            fvcb_str = "  -- "
            af_str = "  --"
            if self.opponent_table is not None:
                stats = self.opponent_table.get(pid)
                if stats.cbets_faced > 0:
                    fvcb_str = f"{stats.fold_to_cbet * 100:>4.0f}%"
                if stats.postflop_calls + stats.postflop_bets_or_raises > 0:
                    af_str = f"{stats.aggression_factor:>4.1f}"
            lines.append(
                f"  {pid:<14} {s['hands']:>6d} "
                f"{s['vpip_pct']:>5.1f}% {s['pfr_pct']:>5.1f}% "
                f"{fvcb_str:>6} {af_str:>5} "
                f"{s['avg_won']:>9.2f}  {s['bb_per_100']:>+7.1f}"
            )
        return "\n".join(lines)


def _serialize_action(rec: ActionRecord) -> dict:
    return {
        "player_id": rec.player_id,
        "action_type": rec.action.type.value,
        "amount": rec.action.amount,
        "street": rec.street.value,
        "pot_before": rec.pot_before,
        "pot_after": rec.pot_after,
    }


def _serialize_award(a: PotAward) -> dict:
    return {
        "amount": a.amount,
        "eligible": list(a.eligible_player_ids),
        "winners": list(a.winner_ids),
        "per_winner": a.per_winner,
    }


def play_one_hand(
    agents: list[Agent],
    starting_stacks: list[float],
    button_index: int,
    blinds: tuple[float, float],
    rng: random.Random,
    hand_id: int,
) -> HandResult:
    """Play exactly one hand; return a HandResult with everything observed."""
    if len(agents) != len(starting_stacks):
        raise ValueError("agents and starting_stacks must align")
    players = [
        Player(id=a.name, stack=s) for a, s in zip(agents, starting_stacks)
    ]
    deck = Deck(rng=rng)
    state = start_hand(players, button_index, deck, blinds)
    agent_by_id = {a.name: a for a in agents}
    starting_by_id = {a.name: s for a, s in zip(agents, starting_stacks)}

    # Capture initial player snapshot
    player_snapshots = [
        {
            "id": p.id,
            "position": p.position.value if p.position else None,
            "hole": [str(c) for c in p.hole_cards] if p.hole_cards else [],
            "starting_stack": starting_by_id[p.id],
        }
        for p in players
    ]

    # Run actions until complete
    max_actions = 200  # safety cap (full 6-max NLHE never gets close)
    for _ in range(max_actions):
        if state.is_complete:
            break
        actor = state.actor()
        view = state.view_for(actor.id)
        action = agent_by_id[actor.id].decide(view)
        # Defensive validation — if a bot returns an illegal action, fold
        try:
            apply_action(state, action)
        except ValueError:
            apply_action(state, _safe_fold())

    awards = award_pots(state)

    # Final stacks
    for snap in player_snapshots:
        snap["ending_stack"] = next(p.stack for p in players if p.id == snap["id"])

    return HandResult(
        hand_id=hand_id,
        button_player_id=players[button_index].id,
        blinds=blinds,
        players=player_snapshots,
        board=[str(c) for c in state.board],
        actions=[_serialize_action(a) for a in state.actions],
        awards=[_serialize_award(a) for a in awards],
        final_pot=sum(a.amount for a in awards),
    )


def _safe_fold():
    from .game import Action, ActionType
    return Action(ActionType.FOLD)


def run_simulation(
    agents: list[Agent],
    num_hands: int,
    starting_stack: float = 100.0,
    blinds: tuple[float, float] = (0.5, 1.0),
    persistent_stacks: bool = False,
    auto_rebuy: bool = True,
    output_path: Optional[Path] = None,
    seed: Optional[int] = None,
    progress_every: int = 0,
    opponent_table: Optional[OpponentTable] = None,
    use_opponent_models: bool = True,
) -> SimStats:
    """Play `num_hands` hands at a table of agents.

    Args:
        agents: list of Agent instances. Each plays a seat.
        num_hands: how many hands to play.
        starting_stack: starting stack per player (in chips). Reset each hand
            unless persistent_stacks=True.
        blinds: (small_blind, big_blind).
        persistent_stacks: if True, stacks carry over hand-to-hand.
        auto_rebuy: if True (and persistent_stacks), busted players reload.
        output_path: if given, append JSONL records here.
        seed: RNG seed for reproducibility.
        progress_every: print a heartbeat every N hands (0 = silent).

    Returns:
        SimStats with aggregate per-player numbers.
    """
    if num_hands <= 0:
        raise ValueError("num_hands must be > 0")
    rng = random.Random(seed) if seed is not None else random.Random()

    # Shared opponent table — created if not passed in. All BotAgents at the
    # table read/write the same one, so they all see each other's stats.
    if opponent_table is None:
        opponent_table = OpponentTable()
    if use_opponent_models:
        for a in agents:
            if isinstance(a, BotAgent):
                a.opponent_table = opponent_table
    else:
        # A/B testing: disable opponent modeling on bots, but still observe
        # actions into the table for stats reporting.
        for a in agents:
            if isinstance(a, BotAgent):
                a.opponent_table = None

    out_file: Optional[TextIO] = None
    if output_path is not None:
        out_file = open(output_path, "w")

    n = len(agents)
    stacks = [starting_stack] * n
    button_index = 0

    # Stats accumulators (per-hand flags, summed across hands)
    per_player: dict[str, dict] = {
        a.name: {
            "hands": 0,
            "vpip_hands": 0,         # hands where they voluntarily put money in (preflop)
            "pfr_hands": 0,          # hands where they raised preflop
            "total_won": 0.0,        # cumulative chips won (positive = profit)
        }
        for a in agents
    }

    try:
        for hand_id in range(num_hands):
            # If auto_rebuy, refresh any busted stacks
            if persistent_stacks and auto_rebuy:
                stacks = [s if s >= blinds[1] else starting_stack for s in stacks]
            elif not persistent_stacks:
                stacks = [starting_stack] * n

            # Skip hand if not enough live players
            live = sum(1 for s in stacks if s > 0)
            if live < 2:
                break

            result = play_one_hand(
                agents=agents, starting_stacks=stacks, button_index=button_index,
                blinds=blinds, rng=rng, hand_id=hand_id,
            )

            # Update per-player stats from result
            for snap in result.players:
                pid = snap["id"]
                per_player[pid]["hands"] += 1
                won = snap["ending_stack"] - snap["starting_stack"]
                per_player[pid]["total_won"] += won
            # Per-hand VPIP/PFR flags
            vpip_this_hand: set[str] = set()
            pfr_this_hand: set[str] = set()
            for action_rec in result.actions:
                if action_rec["street"] != "preflop":
                    continue
                pid = action_rec["player_id"]
                if action_rec["action_type"] in ("call", "bet", "raise"):
                    vpip_this_hand.add(pid)
                if action_rec["action_type"] in ("bet", "raise"):
                    pfr_this_hand.add(pid)
            for pid in vpip_this_hand:
                per_player[pid]["vpip_hands"] += 1
            for pid in pfr_this_hand:
                per_player[pid]["pfr_hands"] += 1

            # Feed the hand into the shared opponent table — bots will see
            # updated stats on the very next hand.
            opponent_table.observe_hand({
                "players": result.players,
                "actions": result.actions,
            })

            if persistent_stacks:
                stacks = [snap["ending_stack"] for snap in result.players]

            if out_file is not None:
                out_file.write(json.dumps({
                    "hand_id": result.hand_id,
                    "button": result.button_player_id,
                    "blinds": list(result.blinds),
                    "players": result.players,
                    "board": result.board,
                    "actions": result.actions,
                    "awards": result.awards,
                    "final_pot": result.final_pot,
                }) + "\n")

            button_index = (button_index + 1) % n

            if progress_every and (hand_id + 1) % progress_every == 0:
                print(f"  ...{hand_id + 1}/{num_hands} hands")
    finally:
        if out_file is not None:
            out_file.close()

    # Build aggregate stats
    stats = SimStats(total_hands=num_hands, opponent_table=opponent_table)
    for pid, s in per_player.items():
        h = max(1, s["hands"])
        vpip_pct = 100.0 * s["vpip_hands"] / h
        pfr_pct = 100.0 * s["pfr_hands"] / h
        avg_won = s["total_won"] / h
        bb_per_100 = (s["total_won"] / blinds[1]) / h * 100
        stats.per_player[pid] = {
            "hands": s["hands"],
            "vpip_pct": vpip_pct,
            "pfr_pct": pfr_pct,
            "avg_won": avg_won,
            "bb_per_100": bb_per_100,
            "total_won": s["total_won"],
        }

    return stats
