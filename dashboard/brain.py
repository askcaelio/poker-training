"""Brain panel renderer — shows the human's reasoning view + action UI.

When it's the human's turn, this panel surfaces equity, pot odds, opponent
read, range, texture, SPR, fold equity — everything the bot would compute
for itself — plus action buttons. The human can play with the same info a
strong bot would have.
"""

from __future__ import annotations

import random
from typing import Optional

from poker.archetype_ranges import call_open_range, open_range, threebet_range
from poker.board_texture import analyze_texture
from poker.equity import equity_vs_random, equity_vs_range
from poker.evaluator import evaluate
from poker.game import (
    Action, ActionType, HandState, Player, Street, legal_actions,
)
from poker.opponent_model import OpponentTable, estimate_fold_equity
from poker.range_model import Range
from poker.range_tracker import RangeTracker

from .render import explainer

HUMAN_ID = "You"


def _hero(state: HandState) -> Player:
    for p in state.players:
        if p.id == HUMAN_ID:
            return p
    raise ValueError("no human player")


def _classify_hand(hole_cards) -> tuple[str, str]:
    """Return (class_string, "AKs"-style label)."""
    c1, c2 = hole_cards
    if c1.rank == c2.rank:
        cls = f"{c1.rank.char}{c2.rank.char}"
    else:
        high, low = (c1, c2) if c1.rank > c2.rank else (c2, c1)
        suited = c1.suit == c2.suit
        cls = f"{high.rank.char}{low.rank.char}{'s' if suited else 'o'}"
    return cls, cls


def render_brain(
    state: HandState,
    opponent_table: Optional[OpponentTable],
    range_tracker: Optional[RangeTracker],
) -> str:
    hero = _hero(state)
    is_your_turn = (
        not state.is_complete
        and state.actor_index == state.players.index(hero)
        and hero.can_act
    )

    if state.is_complete:
        return _render_complete(state)

    if is_your_turn:
        return _render_decision(state, hero, opponent_table, range_tracker)

    # Bot is acting — show "waiting" state
    return _render_waiting(state, hero, opponent_table)


def _render_complete(state: HandState) -> str:
    return (
        '<div class="brain-header">'
        '<div class="role">HAND COMPLETE</div>'
        '<h2>Reviewing the result</h2>'
        '<div class="subtitle">See showdown panel for hands and chip changes.</div>'
        '</div>'
        '<div class="next-hand-cta">'
        '<button class="action-btn primary" hx-post="/next_hand" hx-target="#main" hx-swap="innerHTML">'
        'Next Hand →'
        '</button>'
        '</div>'
    )


def _render_waiting(state: HandState, hero: Player, opponent_table: Optional[OpponentTable]) -> str:
    actor = state.actor()
    return (
        '<div class="brain-header">'
        f'<div class="role">{actor.id} IS THINKING…</div>'
        f'<h2>Waiting on {actor.id}</h2>'
        '<div class="subtitle">Bot decisions resolve automatically.</div>'
        '</div>'
        '<div class="thinking-spinner"><div></div><div></div><div></div></div>'
    )


def _render_decision(
    state: HandState,
    hero: Player,
    opponent_table: Optional[OpponentTable],
    range_tracker: Optional[RangeTracker],
) -> str:
    """The big one — full brain panel for the human's decision."""
    rng = random.Random(2026)

    hole = list(hero.hole_cards) if hero.hole_cards else []
    board = list(state.board)
    pot = state.pot
    to_call = max(0.0, state.current_bet - hero.bet_this_round)
    your_stack = hero.stack
    pos = hero.position.value if hero.position else "?"

    # Identify active opponents
    active_opps = [p for p in state.players if p.id != HUMAN_ID and not p.folded]
    n_opps = max(1, len(active_opps))

    # Hand info
    hand_class_str, hand_label = _classify_hand(hero.hole_cards) if hero.hole_cards else ("", "")
    made_hand_label = "—"
    if board and len(hole) + len(board) >= 5:
        try:
            hs = evaluate(hole, board)
            made_hand_label = hs.category
        except ValueError:
            pass

    # Equity
    eq_random = 0.0
    eq_range = None
    if hole and board:
        eq_random = equity_vs_random(
            hole, board, num_opponents=n_opps, iterations=4000, rng=rng,
        ).equity_pct
        # vs range — only if range_tracker has data and exactly 1 active opp
        if range_tracker and len(active_opps) == 1:
            r = range_tracker.get(active_opps[0].id)
            if r and r.total_combos > 0:
                eq_range = equity_vs_range(
                    hole, r, board, iterations=4000, rng=rng,
                ).equity_pct
    elif hole and not board:
        eq_random = equity_vs_random(
            hole, num_opponents=n_opps, iterations=4000, rng=rng,
        ).equity_pct

    # Pot odds
    if to_call > 0:
        required_eq = 100.0 * to_call / (pot + to_call)
        edge = (eq_range if eq_range is not None else eq_random) - required_eq
    else:
        required_eq = 0.0
        edge = None

    # Texture
    texture_label = "—"
    if board:
        tex = analyze_texture(board)
        if tex.is_monotone:
            texture_label = "MONOTONE · WET"
        elif tex.is_two_tone:
            texture_label = "TWO-TONE · " + ("WET" if tex.is_wet else "DRY")
        elif tex.paired:
            texture_label = "PAIRED"
        else:
            texture_label = "RAINBOW · DRY"

    # SPR
    if active_opps:
        eff_stack = min(your_stack, max(o.stack for o in active_opps))
    else:
        eff_stack = your_stack
    spr = eff_stack / max(pot, 0.01)

    # Opponent: pick the most relevant (the only active opp if heads-up,
    # else the one to most-recently bet/raise this street)
    primary_opp = active_opps[0] if active_opps else None
    primary_opp_stats = opponent_table.get(primary_opp.id) if primary_opp and opponent_table else None

    # Inferred range info
    range_info = None
    if range_tracker and primary_opp:
        r = range_tracker.get(primary_opp.id)
        if r and r.total_combos > 0:
            range_info = {
                "combos": r.total_combos,
                "width_pct": r.width_pct,
            }

    # Fold equity (if facing a check, our potential bluff/value bet's fold equity)
    fold_eq = None
    if primary_opp and opponent_table and to_call == 0:
        fold_eq = estimate_fold_equity(primary_opp.id, opponent_table, "cbet") * 100

    return _build_brain_html(
        hero=hero, hand_label=hand_label, made_hand=made_hand_label,
        eq_random=eq_random, eq_range=eq_range,
        pot=pot, to_call=to_call, required_eq=required_eq, edge=edge,
        texture_label=texture_label, spr=spr, eff_stack=eff_stack,
        primary_opp=primary_opp, primary_opp_stats=primary_opp_stats,
        range_info=range_info, fold_eq=fold_eq,
        state=state, pos=pos,
    )


def _build_brain_html(*, hero, hand_label, made_hand, eq_random, eq_range,
                      pot, to_call, required_eq, edge, texture_label, spr,
                      eff_stack, primary_opp, primary_opp_stats, range_info,
                      fold_eq, state, pos) -> str:
    eq_main = eq_range if eq_range is not None else eq_random
    eq_main_color = "positive" if eq_main >= 55 else ("negative" if eq_main < 35 else "")
    edge_str = ""
    if edge is not None:
        sign = "+" if edge >= 0 else ""
        color = "positive" if edge > 5 else ("negative" if edge < -5 else "")
        edge_str = f'<span class="number {color}">{sign}{edge:.1f}pp</span>'

    range_block = ""
    if range_info:
        range_block = f'''
<div class="module">
  <div class="module-header"><span class="module-title">Inferred Range</span></div>
  <div class="stat-row"><span class="label">Combos remaining</span><span></span><span class="number highlight">{range_info["combos"]:.0f}</span></div>
  <div class="stat-row"><span class="label">% of all hands</span><span></span><span class="number">{range_info["width_pct"]:.1f}%</span></div>
  {explainer("range narrowing", "<p>Each action a player takes <em>narrows</em> what they could plausibly hold. We track the inferred range live as the hand progresses — calls and raises remove different sets of hands.</p><p>The <em>combos remaining</em> shows how many specific 2-card combinations are still in their range, out of 1326 possible. By the river, a tight range can shrink to a handful of combos.</p>", "ranges")}
</div>
'''

    opp_block = ""
    if primary_opp_stats:
        s = primary_opp_stats
        opp_block = f'''
<div class="module">
  <div class="module-header"><span class="module-title">Opponent · {primary_opp.id}</span></div>
  <div class="stat-row"><span class="label">Hands seen</span><span></span><span class="number">{s.hands_played}</span></div>
  <div class="stat-row"><span class="label">VPIP</span><span></span><span class="number">{s.vpip * 100:.0f}%</span></div>
  <div class="stat-row"><span class="label">PFR</span><span></span><span class="number">{s.pfr * 100:.0f}%</span></div>
  <div class="stat-row"><span class="label">Fold to cbet</span><span></span><span class="number {"highlight" if s.cbets_faced > 0 else ""}">{s.fold_to_cbet * 100:.0f}%</span></div>
  <div class="stat-row"><span class="label">Aggression</span><span></span><span class="number">{s.aggression_factor:.1f}</span></div>
  {explainer("opponent stats", "<p>Four numbers tell most of the story: <em>VPIP</em> (how often they enter pots), <em>PFR</em> (how often they raise preflop), <em>Fold to C-bet</em> (how often they fold the flop after calling preflop), and <em>Aggression Factor</em> (bets+raises per call).</p><p>VPIP-PFR gap shows passivity — a player at 30%/5% calls a lot but rarely raises (a station). High aggression factor + high VPIP = LAG/maniac.</p>", "stats")}
</div>
'''

    fold_eq_block = ""
    if fold_eq is not None:
        fold_eq_block = f'''
<div class="module">
  <div class="module-header"><span class="module-title">Fold Equity</span></div>
  <div class="stat-row"><span class="label">If you bet c-bet</span><span></span><span class="number highlight">{fold_eq:.0f}% fold</span></div>
  {explainer("fold equity", "<p><em>Fold equity</em> is the probability your bet causes villain to fold. It's the other half of EV — pure equity assumes you go to showdown, but most pots end before that.</p><p>EV(bet) = fold_eq × pot + (1−fold_eq) × (eq × (pot+2bet) − bet). When fold equity is high, even weak hands become profitable bets (bluffs). Vs a calling station, only strong hands should bet for value — bluffs don't work.</p>", "foldeq")}
</div>
'''

    pot_odds_block = ""
    if to_call > 0:
        pot_odds_block = f'''
<div class="module">
  <div class="module-header"><span class="module-title">Pot Odds</span></div>
  <div class="figures">
    <div class="figure"><span class="v">${pot:.0f}</span><span class="l">Pot</span></div>
    <div class="figure"><span class="v">${to_call:.2f}</span><span class="l">To Call</span></div>
    <div class="figure"><span class="v">{required_eq:.0f}%</span><span class="l">You Need</span></div>
  </div>
  <div class="stat-row"><span class="label">Required eq</span><span></span><span class="number">{required_eq:.1f}%</span></div>
  <div class="stat-row"><span class="label">Actual eq</span><span></span><span class="number {eq_main_color}">{eq_main:.1f}%</span></div>
  <div class="stat-row"><span class="label">Edge</span><span></span>{edge_str}</div>
  {explainer("pot odds", "<p><em>Pot odds</em> are the price the pot is offering. If villain bets $10 into a $30 pot, you're being asked to call $10 to win $40 — you need 25% equity to break even.</p><p><em>Edge</em> is your equity advantage over the break-even threshold. Positive edge = the call is +EV.</p>", "potodds")}
</div>
'''

    eq_block = f'''
<div class="module">
  <div class="module-header"><span class="module-title">Equity</span></div>
  <div class="stat-row">
    <span class="label">vs Random</span>
    <div class="bar"><div class="bar-fill" style="width: {min(100, round(eq_random))}%"></div></div>
    <span class="number">{eq_random:.1f}%</span>
  </div>
  {f"""<div class="stat-row">
    <span class="label">vs Their Range</span>
    <div class="bar"><div class="bar-fill jade" style="width: {min(100, round(eq_range))}%"></div></div>
    <span class="number positive">{eq_range:.1f}%</span>
  </div>""" if eq_range is not None else ""}
  {explainer("equity", "<p><em>Equity</em> is your probability of winning at showdown if all remaining cards come out. We compute it by Monte Carlo: deal thousands of possible runouts, count wins.</p><p><em>vs Random</em> assumes villain has any two cards. <em>vs Their Range</em> uses the inferred range based on observed actions — usually more accurate.</p>", "equity")}
</div>
'''

    # Action buttons
    actions_html = _render_action_ui(state, hero)

    return (
        '<div class="brain-header">'
        f'<div class="role">YOUR TURN · {pos} · {hand_label}</div>'
        '<h2>Your move</h2>'
        f'<div class="subtitle">{state.street.value.title()} · stack ${hero.stack:.2f} · pot ${pot:.2f}</div>'
        '</div>'
        f'''
<div class="module">
  <div class="module-header"><span class="module-title">My Hand</span></div>
  <div class="stat-row"><span class="label">Class</span><span></span><span class="number highlight">{hand_label}</span></div>
  <div class="stat-row"><span class="label">Made</span><span></span><span class="number">{made_hand}</span></div>
</div>
{eq_block}
{pot_odds_block}
{opp_block}
{range_block}
'''
        f'''
<div class="module">
  <div class="module-header"><span class="module-title">Board &amp; SPR</span></div>
  <div class="stat-row"><span class="label">Texture</span><span></span><span class="number highlight">{texture_label}</span></div>
  <div class="stat-row"><span class="label">Effective stack</span><span></span><span class="number">${eff_stack:.2f}</span></div>
  <div class="stat-row"><span class="label">SPR</span><span></span><span class="number">{spr:.1f}</span></div>
  {explainer("SPR &amp; texture", "<p><em>SPR</em> (Stack-to-Pot Ratio) tells you how committed you are. SPR &lt; 2 = play for stacks. SPR &gt; 6 = deep, careful with one pair.</p><p><em>Texture</em> shapes optimal sizing. Wet boards (draws everywhere) → bigger bets to charge them. Dry boards → smaller probing bets.</p>", "spr")}
</div>
{fold_eq_block}
{actions_html}
'''
    )


def _render_action_ui(state: HandState, hero: Player) -> str:
    legal = legal_actions(state)
    to_call = max(0.0, state.current_bet - hero.bet_this_round)
    pot = state.pot
    min_raise_inc = max(state.last_raise_size, state.blinds[1])

    bet_default = max(round(pot * 0.66, 2), state.blinds[1])
    raise_default_total = state.current_bet + min_raise_inc
    raise_default_contribution = raise_default_total - hero.bet_this_round
    pot_3x_total = state.current_bet + pot
    pot_3x_contribution = pot_3x_total - hero.bet_this_round

    buttons = []
    buttons.append('<div class="action-row">')
    if ActionType.FOLD in legal:
        buttons.append(_btn("FOLD", "fold", classes="negative"))
    if ActionType.CHECK in legal:
        buttons.append(_btn("CHECK", "check", classes=""))
    if ActionType.CALL in legal:
        buttons.append(_btn(f"CALL ${to_call:.2f}", "call", amount=to_call, classes=""))
    buttons.append('</div>')

    if ActionType.BET in legal or ActionType.RAISE in legal:
        action_type = "raise" if ActionType.RAISE in legal else "bet"
        default_amount = raise_default_contribution if action_type == "raise" else bet_default
        max_amount = hero.stack
        buttons.append('<div class="action-row sizing-row">')
        buttons.append(
            f'<form class="bet-form" hx-post="/act" hx-target="#main" hx-swap="innerHTML">'
            f'<input type="hidden" name="action_type" value="{action_type}">'
            f'<input class="bet-input" type="number" step="0.5" name="amount" min="{state.blinds[1]}" max="{max_amount:.2f}" value="{default_amount:.2f}">'
            f'<button class="action-btn primary" type="submit">{action_type.upper()}</button>'
            '</form>'
        )

        # Quick-size buttons
        if action_type == "bet":
            quick_sizes = [
                ("1/3 pot", round(pot * 0.33, 2)),
                ("2/3 pot", round(pot * 0.66, 2)),
                ("Pot", round(pot, 2)),
            ]
        else:
            quick_sizes = [
                ("Min", round(raise_default_contribution, 2)),
                ("Pot", round(pot_3x_contribution, 2)),
                ("All-in", round(hero.stack, 2)),
            ]
        for label, amt in quick_sizes:
            if amt > 0 and amt <= hero.stack:
                buttons.append(
                    f'<button class="action-btn quick" type="button" onclick="document.querySelector(\'.bet-input\').value={amt:.2f}">{label}</button>'
                )
        buttons.append('</div>')

    return '<div class="action-area">' + "".join(buttons) + '</div>'


def _btn(label: str, action_type: str, *, amount: float = 0.0, classes: str = "") -> str:
    return (
        f'<form class="action-form" hx-post="/act" hx-target="#main" hx-swap="innerHTML">'
        f'<input type="hidden" name="action_type" value="{action_type}">'
        f'<input type="hidden" name="amount" value="{amount}">'
        f'<button class="action-btn {classes}" type="submit">{label}</button>'
        '</form>'
    )
