"""Play poker against the bots, with the dashboard shown for every decision.

This is the interactive trainer: you make the calls, the dashboard tells you
what equity you actually have so you can build intuition.

Usage:
    uv run python scripts/play.py [num_hands]

Default: 50 hands at a 6-max table with one of each bot archetype.
"""

import random
import sys
from typing import Optional

from poker.agents import (
    Agent, BotAgent, make_lag, make_maniac, make_nit, make_station, make_tag,
)
from poker.cards import Deck
from poker.dashboard import build, render
from poker.game import (
    Action, ActionType, HandState, HandView, Player, apply_action, award_pots,
    legal_actions, start_hand,
)


class HumanAgent(Agent):
    """Prompt the human via stdin, with the dashboard shown each decision."""

    def __init__(self, name: str = "You"):
        self.name = name

    def decide(self, view: HandView) -> Action:
        # We don't have the full HandState here, just the view. The dashboard
        # uses what's in the view. For legal_actions we need state — but the
        # caller in play_one_hand_interactive will pass that separately.
        raise NotImplementedError(
            "HumanAgent.decide is called via the interactive driver, not the runner."
        )


def _print_dashboard(view: HandView, num_active_opponents: int) -> None:
    """Show the dashboard for the current decision."""
    d = build(
        hole=list(view.your_hole),
        board=list(view.board),
        num_opponents=num_active_opponents,
        pot=view.pot if view.to_call > 0 else None,
        call=view.to_call if view.to_call > 0 else None,
        iterations=5_000,
    )
    print(render(d))


def _prompt_human(view: HandView, legal: list[ActionType]) -> Action:
    """Show a menu of legal actions and read a choice."""
    print(f"\nYour stack: {view.your_stack:.2f}   Pot: {view.pot:.2f}   "
          f"To call: {view.to_call:.2f}")
    options: list[tuple[str, str, ActionType]] = []
    if ActionType.FOLD in legal:
        options.append(("f", "Fold", ActionType.FOLD))
    if ActionType.CHECK in legal:
        options.append(("k", "Check", ActionType.CHECK))
    if ActionType.CALL in legal:
        options.append(("c", f"Call {view.to_call:.2f}", ActionType.CALL))
    if ActionType.BET in legal:
        options.append(("b", "Bet (then enter amount)", ActionType.BET))
    if ActionType.RAISE in legal:
        options.append(("r", "Raise (then enter amount)", ActionType.RAISE))

    menu = "  ".join(f"[{key}] {label}" for key, label, _ in options)
    print(f"Action: {menu}")

    while True:
        choice = input("> ").strip().lower()
        for key, _, atype in options:
            if choice.startswith(key):
                if atype == ActionType.CALL:
                    return Action(ActionType.CALL, amount=view.to_call)
                if atype in (ActionType.FOLD, ActionType.CHECK):
                    return Action(atype)
                # Need amount
                while True:
                    raw = input(f"  Amount (min raise = {view.min_raise:.2f}, "
                                f"max = {view.your_stack:.2f}): ").strip()
                    try:
                        amount = float(raw)
                        return Action(atype, amount=amount)
                    except ValueError:
                        print("  Invalid number; try again.")
        print(f"Unknown option {choice!r}; try one of: {menu}")


def play_one_hand_interactive(
    agents: list[Agent],
    starting_stacks: list[float],
    button_index: int,
    blinds: tuple[float, float],
    rng: random.Random,
    human_id: str,
) -> dict:
    """One hand with the human in seat `human_id`, bots elsewhere."""
    players = [
        Player(id=a.name, stack=s) for a, s in zip(agents, starting_stacks)
    ]
    deck = Deck(rng=rng)
    state = start_hand(players, button_index, deck, blinds)
    agent_by_id = {a.name: a for a in agents}

    print("\n" + "█" * 60)
    you = next(p for p in players if p.id == human_id)
    print(f" New hand. You have: "
          f"{' '.join(c.pretty() for c in you.hole_cards or [])}   "
          f"Position: {you.position.value if you.position else '?'}")
    print(f" Button: {players[button_index].id}")
    print("█" * 60)

    last_street = None
    while not state.is_complete:
        actor = state.actor()
        view = state.view_for(actor.id)

        if view.street != last_street and view.board:
            board_str = " ".join(c.pretty() for c in view.board)
            print(f"\n--- {view.street.value.upper()} --- {board_str}")
            last_street = view.street

        if actor.id == human_id:
            num_opps = view.num_active_opponents
            _print_dashboard(view, num_opps)
            legal = legal_actions(state)
            action = _prompt_human(view, legal)
            try:
                apply_action(state, action)
            except ValueError as e:
                print(f"  ⚠ {e}. Try again.")
                continue
            print(f"  → You {action}")
        else:
            action = agent_by_id[actor.id].decide(view)
            try:
                apply_action(state, action)
            except ValueError:
                apply_action(state, Action(ActionType.FOLD))
            print(f"  → {actor.id} ({type(agent_by_id[actor.id]).__name__}): {action}")

    print("\n" + "─" * 60)
    awards = award_pots(state)
    for a in awards:
        winners = ", ".join(a.winner_ids)
        print(f"  Pot {a.amount:.2f} → {winners} ({a.per_winner:.2f} each)")
    return {
        "human_won": next(
            (a.per_winner for a in awards if human_id in a.winner_ids), 0.0
        ),
    }


def main() -> None:
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rng = random.Random()

    human = HumanAgent("You")
    agents: list[Agent] = [
        human,
        make_nit("Nit"),
        make_tag("TAG"),
        make_lag("LAG"),
        make_maniac("Maniac"),
        make_station("Station"),
    ]

    print("=" * 60)
    print("  poker-training: interactive trainer")
    print(f"  6-max NLHE cash, blinds 0.5/1.0, starting stack 100bb")
    print(f"  Playing {num_hands} hands. Ctrl+C to quit anytime.")
    print("=" * 60)

    stacks = [100.0] * len(agents)
    button_index = 0

    try:
        for hand_id in range(num_hands):
            # auto-rebuy busted players (cash-game behavior)
            stacks = [s if s > 0 else 100.0 for s in stacks]
            result = play_one_hand_interactive(
                agents=agents, starting_stacks=stacks, button_index=button_index,
                blinds=(0.5, 1.0), rng=rng, human_id="You",
            )
            # We don't track stacks across hands unless we want to (cash always rebuy).
            # For "running session" mode, swap in persistent stacks here.
            button_index = (button_index + 1) % len(agents)
    except (KeyboardInterrupt, EOFError):
        print("\n\n(stopped)")


if __name__ == "__main__":
    main()
