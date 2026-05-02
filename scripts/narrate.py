"""Play one hand with full narration — bot reasoning + dashboard from one POV.

Designed for understanding what the engine is doing on every decision.
Pick a POV player (default TAG_1), see their dashboard each time it's their
turn, and see annotated one-liners for every other bot's action.

Usage:
    uv run python scripts/narrate.py [seed] [pov_id]

Defaults: seed=2026, pov=TAG_1.
"""

import random
import sys

from poker.agents import (
    Agent, BotAgent, hand_class,
    NIT_RANGE, TAG_RANGE, LAG_RANGE, MANIAC_RANGE, STATION_RANGE,
    make_lag, make_maniac, make_nit, make_station, make_tag,
)
from poker.cards import Deck
from poker.dashboard import build, render
from poker.equity import equity_vs_random
from poker.game import (
    Action, ActionType, HandState, HandView, Player, Position, Street,
    apply_action, award_pots, start_hand,
)


_RANGE_NAMES: list[tuple[str, set[str]]] = [
    ("NIT", NIT_RANGE),
    ("TAG", TAG_RANGE),
    ("LAG", LAG_RANGE),
    ("MANIAC", MANIAC_RANGE),
    ("STATION", STATION_RANGE),
]


def _why_preflop(agent_name: str, archetype: str, hand: str, action: Action,
                 view: HandView) -> str:
    """One-line annotation explaining a preflop decision."""
    in_range_label = next((name for name, r in _RANGE_NAMES if hand in r), "outside any range")
    facing_raise = view.current_bet > 1.0
    if action.type == ActionType.FOLD:
        if hand in {"AA", "KK", "QQ", "AKs", "AKo"}:
            return f"folds premium {hand}?? (config quirk)"
        return f"{hand} not in {archetype} range (range = {in_range_label})"
    if action.type == ActionType.CHECK:
        return f"{hand} checks BB option (free)"
    if action.type == ActionType.CALL:
        if facing_raise:
            return f"{hand} flat-calls the raise (in range, not strong enough to 3-bet)"
        return f"{hand} limps (Station-style: wants to see flop cheap)"
    if action.type == ActionType.RAISE:
        if facing_raise:
            return f"{hand} 3-bets (premium hand, in 3-bet range)"
        return f"{hand} opens to {action.amount:.1f} BB (in {archetype} opening range)"
    return ""


def _why_postflop(agent: BotAgent, view: HandView, action: Action) -> str:
    """One-line annotation explaining a postflop decision (with equity + fold-eq)."""
    n_opps = max(1, view.num_active_opponents)
    eq = equity_vs_random(
        hole=list(view.your_hole), board=list(view.board),
        num_opponents=n_opps, iterations=1500,
        rng=random.Random(0),  # deterministic for narration
    ).equity_pct
    if view.to_call > 0:
        req = 100 * view.to_call / (view.pot + view.to_call)
        odds = f"need {req:.0f}%, has {eq:.0f}%"
    else:
        odds = f"equity {eq:.0f}%"

    # If the bot uses opponent models, also show its fold-equity estimate
    fe_note = ""
    if agent.opponent_table is not None and agent.config.uses_opponent_model:
        from poker.opponent_model import estimate_fold_equity
        active = [o for o in view.others if not o.folded]
        if active:
            # Use min fold-eq across opponents (matches bot's own logic)
            fold_eqs = [
                estimate_fold_equity(o.id, agent.opponent_table, "cbet")
                for o in active
            ]
            min_fe = min(fold_eqs)
            fe_note = f", min fold-eq {min_fe * 100:.0f}%"

    if action.type == ActionType.FOLD:
        return f"folds — {odds}{fe_note}"
    if action.type == ActionType.CHECK:
        return f"checks — {odds}{fe_note}, EV(check) > EV(bet)"
    if action.type == ActionType.CALL:
        return f"calls — {odds}{fe_note}, +EV"
    if action.type == ActionType.BET:
        return f"bets {action.amount:.1f} — {odds}{fe_note}, +EV(bet)"
    if action.type == ActionType.RAISE:
        return f"raises ~{view.your_bet_this_round + action.amount:.1f} — {odds}{fe_note}"
    return ""


def narrate_hand(agents: list[Agent], pov_id: str, button_index: int,
                 blinds: tuple[float, float], rng: random.Random) -> None:
    players = [Player(id=a.name, stack=100.0) for a in agents]
    deck = Deck(rng=rng)
    state = start_hand(players, button_index, deck, blinds)
    agent_by_id = {a.name: a for a in agents}

    print()
    print("█" * 64)
    print(f" HAND: 6-max NLHE cash, blinds {blinds[0]}/{blinds[1]}, stacks 100 BB")
    print(f" Following along with: {pov_id}")
    print("█" * 64)
    print()
    print(" Seats (button = " + players[button_index].id + "):")
    for p in players:
        marker = "  ← POV" if p.id == pov_id else ""
        cards = " ".join(c.pretty() for c in p.hole_cards or [])
        # Hide other players' cards from the user's view (just show ?? ??)
        if p.id != pov_id:
            cards_disp = "?? ??"
        else:
            cards_disp = cards
        print(f"   {p.id:<10} [{p.position.value:>3}]  100.00 BB   {cards_disp}{marker}")
    print(f"\n Blinds posted: SB {blinds[0]}, BB {blinds[1]}. Pot {state.pot:.2f}.")

    last_street: Street | None = None
    while not state.is_complete:
        actor = state.actor()
        view = state.view_for(actor.id)

        # Street header
        if view.street != last_street:
            board = " ".join(c.pretty() for c in view.board) if view.board else "(none)"
            print(f"\n══ {view.street.value.upper()} ══   Board: {board}   Pot: {view.pot:.2f}")
            last_street = view.street

        # Show POV dashboard if it's our turn
        if actor.id == pov_id:
            n_opps = max(1, view.num_active_opponents)
            print()
            print(f"  ┌─── {pov_id}'s dashboard ───")
            d = build(
                hole=list(view.your_hole),
                board=list(view.board),
                num_opponents=n_opps,
                pot=view.pot if view.to_call > 0 else None,
                call=view.to_call if view.to_call > 0 else None,
                iterations=5000,
            )
            for line in render(d).splitlines():
                print(f"  │ {line}")
            print(f"  └─")

        # Get the action
        action = agent_by_id[actor.id].decide(view)
        try:
            apply_action(state, action)
        except ValueError:
            apply_action(state, Action(ActionType.FOLD))

        # Annotate
        if isinstance(agent_by_id[actor.id], BotAgent):
            archetype = agent_by_id[actor.id].config.name
            hand = hand_class(*actor.hole_cards) if actor.hole_cards else ""
            if view.street == Street.PREFLOP:
                why = _why_preflop(actor.id, archetype, hand, action, view)
            else:
                why = _why_postflop(agent_by_id[actor.id], view, action)
            tag = f"[{archetype}]"
        else:
            why = ""
            tag = ""

        action_str = str(action)
        prefix = "  → " if actor.id != pov_id else "  ▶▶ "
        line = f"{prefix}{actor.id:<10} {tag:<10} {action_str:<14}  {why}"
        print(line)

    # Showdown
    print(f"\n══ SHOWDOWN ══")
    if state.num_in_hand > 1:
        print(" Final board:", " ".join(c.pretty() for c in state.board))
        for p in players:
            if p.in_hand:
                print(f"   {p.id:<10} {' '.join(c.pretty() for c in p.hole_cards or [])}")
    else:
        winner = next(p for p in players if p.in_hand)
        print(f" {winner.id} wins uncontested (everyone else folded).")

    print()
    awards = award_pots(state)
    for a in awards:
        winners = ", ".join(a.winner_ids)
        print(f" Pot ${a.amount:.2f} → {winners}  (${a.per_winner:.2f} each)")

    # Final stacks
    print("\n Stack changes:")
    for p in players:
        delta = p.stack - 100.0
        sign = "+" if delta >= 0 else ""
        marker = "  ← POV" if p.id == pov_id else ""
        print(f"   {p.id:<10} {p.stack:>7.2f} BB ({sign}{delta:.2f}){marker}")


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    pov_id = sys.argv[2] if len(sys.argv) > 2 else "TAG_1"

    rng = random.Random(seed)
    agents = [
        make_nit("Nit_1"),
        make_tag("TAG_1"),
        make_tag("TAG_2"),
        make_lag("LAG_1"),
        make_maniac("Maniac_1"),
        make_station("Station_1"),
    ]
    # Fix bot RNGs to seed too, so the run is fully reproducible
    for i, a in enumerate(agents):
        if isinstance(a, BotAgent):
            a.rng = random.Random(seed + i + 1)

    narrate_hand(agents, pov_id=pov_id, button_index=0,
                 blinds=(0.5, 1.0), rng=rng)


if __name__ == "__main__":
    main()
