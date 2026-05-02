"""Generate a static HTML mockup of the bot-perspective dashboard.

Sets up one specific hand state, computes everything the bot reasons about,
and renders to dashboard/index.html. Open it in a browser:

    uv run python scripts/dashboard_mockup.py
    open dashboard/index.html

The HTML is self-contained: Google Fonts + inline CSS, no build step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from poker.archetype_ranges import call_open_range, open_range
from poker.board_texture import analyze_texture
from poker.cards import Card, Rank, Suit, parse_cards
from poker.equity import equity_vs_random, equity_vs_range
from poker.evaluator import evaluate
from poker.opponent_model import OpponentStats
from poker.range_model import Range


# ─── Hand setup (the scenario being visualized) ──────────────────────────────

HERO_HOLE = parse_cards("As Ks")
BOARD = parse_cards("Kh 7c 2d")          # top pair top kicker, dry
POT = 6.0                                 # pot before any flop bet
HERO_STACK = 97.5                         # hero raised 2.5bb preflop
VILLAIN_STACK = 97.0                      # bb defended

# Synthetic opponent stats for the BB defender (after 47 hands of observation)
VILLAIN_STATS = OpponentStats(
    hands_played=47,
    vpip_hands=9,         # 9/47 = 19%
    pfr_hands=5,          # 5/47 = 11%
    cbets_faced=15,
    cbets_folded_to=7,    # 7/15 = 47% fold to cbet
    threebets_faced=2,
    threebets_folded_to=1,
    postflop_bets_or_raises=21,
    postflop_calls=10,    # AF = 21/10 = 2.1
)


# ─── Compute everything the bot would reason about ──────────────────────────

@dataclass
class Snapshot:
    eq_vs_random: float
    eq_vs_range: float
    pot_odds_required_pct: float
    bot_bet_size: float
    spr: float
    texture_label: str
    texture_is_dry: bool
    texture_is_wet: bool
    villain_archetype: str
    villain_range_combos_initial: float
    villain_range_combos_now: float
    decision: str
    decision_reason: str
    fold_equity_estimate: float


def compute() -> Snapshot:
    rng = random.Random(2026)

    # Texture
    tex = analyze_texture(BOARD)
    texture_label = "RAINBOW · DRY"
    if tex.is_monotone:
        texture_label = "MONOTONE · WET"
    elif tex.is_two_tone:
        texture_label = "TWO-TONE · " + ("WET" if tex.is_wet else "DRY")

    # Equity vs random (1 opp)
    eq_random = equity_vs_random(
        list(HERO_HOLE), list(BOARD), num_opponents=1,
        iterations=10_000, rng=rng,
    ).equity_pct

    # Equity vs villain's narrowed range (TAG who called preflop)
    villain_range_initial = open_range("TAG")
    villain_range_now = call_open_range("TAG")
    eq_range = equity_vs_range(
        list(HERO_HOLE), villain_range_now, list(BOARD),
        iterations=10_000, rng=rng,
    ).equity_pct

    # Hero's c-bet sizing on a dry board
    cbet_pct = 0.66 * 0.7   # TAG base × dry-board adjustment
    bet_size = POT * cbet_pct

    # Pot odds the villain would face if they call
    pot_odds_required_pct = 100.0 * bet_size / (POT + 2 * bet_size)

    # SPR
    effective_stack = min(HERO_STACK, VILLAIN_STACK)
    spr = effective_stack / POT

    # Fold-equity estimate (from villain's observed FvCB)
    fold_equity = VILLAIN_STATS.fold_to_cbet * 100

    # Decision: top pair top kicker on dry board, +EV cbet for value+protection
    decision = f"BET ${bet_size:.2f}"
    decision_reason = (
        "Top pair top kicker on a dry board. Equity vs villain's calling range "
        "is high; villain folds 47% of the time to a c-bet, so this bet "
        "captures fold equity from weaker hands and value from worse pairs that "
        "call. Smaller sizing on dry boards extracts thin value efficiently."
    )

    return Snapshot(
        eq_vs_random=eq_random,
        eq_vs_range=eq_range,
        pot_odds_required_pct=pot_odds_required_pct,
        bot_bet_size=bet_size,
        spr=spr,
        texture_label=texture_label,
        texture_is_dry=tex.is_dry,
        texture_is_wet=tex.is_wet,
        villain_archetype="TAG",
        villain_range_combos_initial=villain_range_initial.total_combos,
        villain_range_combos_now=villain_range_now.total_combos,
        decision=decision,
        decision_reason=decision_reason,
        fold_equity_estimate=fold_equity,
    )


# ─── HTML rendering ──────────────────────────────────────────────────────────

def card_html(card: Card, *, face_down: bool = False, classes: str = "") -> str:
    """Render a single playing card."""
    if face_down:
        return f"""
<div class="card card-back {classes}">
  <div class="card-back-pattern"></div>
</div>
""".strip()

    suit_glyph = card.suit.glyph
    rank_str = card.rank.char
    suit_color_class = "suit-red" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "suit-black"

    return f"""
<div class="card {suit_color_class} {classes}">
  <div class="card-corner top">
    <span class="card-rank">{rank_str}</span>
    <span class="card-suit">{suit_glyph}</span>
  </div>
  <div class="card-pip">{suit_glyph}</div>
  <div class="card-corner bottom">
    <span class="card-rank">{rank_str}</span>
    <span class="card-suit">{suit_glyph}</span>
  </div>
</div>
""".strip()


def explainer(title: str, body: str, anchor: str) -> str:
    """A collapsible <details> explainer with serif-typeset body."""
    return f"""
<details class="explainer" id="exp-{anchor}">
  <summary>
    <span class="explainer-icon">?</span>
    <span class="explainer-label">What is {title}?</span>
  </summary>
  <div class="explainer-body">{body}</div>
</details>
""".strip()


def render(snapshot: Snapshot) -> str:
    s = snapshot
    eq_random_bar_width = round(s.eq_vs_random)
    eq_range_bar_width = round(s.eq_vs_range)
    required_bar_width = round(s.pot_odds_required_pct)
    villain_range_pct_initial = 100.0 * s.villain_range_combos_initial / 1326
    villain_range_pct_now = 100.0 * s.villain_range_combos_now / 1326

    hero_cards_html = "\n".join(card_html(c) for c in HERO_HOLE)
    board_cards_html = "\n".join(
        card_html(c, classes="board-card") for c in BOARD
    )

    # Opponent face-down "dealt" cards for the table view
    opponent_cards = card_html(parse_cards("2c")[0], face_down=True, classes="mini") + \
                     card_html(parse_cards("3c")[0], face_down=True, classes="mini")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Poker Training — Bot's-Eye View</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #0e1410;
    --ink-2: #161e18;
    --paper: #f5efe2;
    --paper-2: #efe7d2;
    --rule: #2a3329;
    --rule-soft: rgba(245, 239, 226, 0.10);
    --suit-red: #b3232a;
    --suit-black: #1a1a1a;
    --gold: #c79a4a;
    --jade: #6f9173;
    --sienna: #b76847;
    --serif: 'Playfair Display', 'Iowan Old Style', Georgia, serif;
    --serif-body: 'Crimson Pro', 'Iowan Old Style', Georgia, serif;
    --mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background: var(--ink);
    color: var(--paper);
    font-family: var(--serif-body);
    font-size: 17px;
    line-height: 1.5;
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse at 30% 20%, rgba(111, 145, 115, 0.08), transparent 50%),
      radial-gradient(ellipse at 80% 80%, rgba(199, 154, 74, 0.04), transparent 50%);
  }}

  /* ─── Header ─── */
  header {{
    border-bottom: 1px solid var(--rule);
    padding: 1.6rem 2.5rem 1.4rem;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }}
  .brand {{
    font-family: var(--serif);
    font-style: italic;
    font-weight: 500;
    font-size: 1.3rem;
    letter-spacing: 0.5px;
  }}
  .brand .accent {{ color: var(--gold); font-style: normal; }}
  .meta {{
    font-family: var(--mono);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(245, 239, 226, 0.55);
  }}

  /* ─── Layout ─── */
  .grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    gap: 0;
    min-height: calc(100vh - 80px);
  }}
  .table-pane {{
    padding: 3rem 2.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-right: 1px solid var(--rule);
    position: relative;
  }}
  .brain-pane {{
    padding: 2rem 2.5rem 4rem;
    overflow-y: auto;
    max-height: calc(100vh - 80px);
  }}

  /* ─── Table ─── */
  .table-frame {{
    width: 100%;
    max-width: 540px;
    aspect-ratio: 1.6 / 1;
    background: radial-gradient(ellipse at center, #1f2a1f 0%, #131b13 70%);
    border-radius: 50% / 60%;
    border: 8px solid #4a3a1f;
    box-shadow:
      inset 0 0 60px rgba(0, 0, 0, 0.5),
      0 30px 60px rgba(0, 0, 0, 0.4);
    position: relative;
    margin-bottom: 1.5rem;
  }}
  .seat {{
    position: absolute;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.7);
  }}
  .seat-name {{ display: block; margin-top: 4px; font-size: 0.65rem; opacity: 0.7; }}
  .seat .stack {{ display: block; color: var(--gold); font-weight: 700; font-size: 0.78rem; }}
  .seat-cards {{ display: flex; gap: 2px; justify-content: center; margin-bottom: 4px; }}

  .seat-utg {{ top: -10px; left: 50%; transform: translateX(-50%); }}
  .seat-mp {{ top: 25%; right: -30px; }}
  .seat-co {{ bottom: 25%; right: -30px; }}
  .seat-btn {{ bottom: -10px; left: 50%; transform: translateX(-50%); }}
  .seat-sb {{ bottom: 25%; left: -30px; }}
  .seat-bb {{ top: 25%; left: -30px; }}
  .seat.folded {{ opacity: 0.35; }}
  .seat.acting {{ color: var(--gold); }}
  .seat.acting::after {{
    content: '';
    display: block;
    width: 6px; height: 6px;
    background: var(--gold);
    border-radius: 50%;
    margin: 4px auto 0;
    box-shadow: 0 0 12px var(--gold);
    animation: pulse 1.6s ease-in-out infinite;
  }}
  @keyframes pulse {{ 50% {{ opacity: 0.4; transform: scale(0.8); }} }}

  .pot-display {{
    position: absolute;
    top: 38%;
    left: 50%;
    transform: translateX(-50%);
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.6);
    text-align: center;
  }}
  .pot-display .amount {{
    display: block;
    font-family: var(--serif);
    font-style: italic;
    font-size: 2rem;
    color: var(--gold);
    margin-top: 4px;
    letter-spacing: 0;
  }}

  .board {{
    position: absolute;
    top: 56%;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 6px;
  }}

  .hero-area {{
    text-align: center;
    margin-top: 0.5rem;
  }}
  .hero-area .label {{
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.6rem;
  }}
  .hero-cards {{ display: flex; gap: 8px; justify-content: center; }}
  .hero-info {{
    margin-top: 0.8rem;
    display: flex;
    gap: 2rem;
    justify-content: center;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: rgba(245, 239, 226, 0.7);
  }}
  .hero-info .stack {{ color: var(--gold); }}

  /* ─── Cards ─── */
  .card {{
    width: 56px;
    height: 80px;
    background: var(--paper);
    border-radius: 6px;
    box-shadow:
      0 2px 6px rgba(0, 0, 0, 0.4),
      inset 0 0 0 1px rgba(0, 0, 0, 0.05);
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 4px 6px;
    font-family: 'Iowan Old Style', Georgia, serif;
    font-weight: 600;
    background-image: linear-gradient(135deg, rgba(0,0,0,0.02) 0%, transparent 50%);
  }}
  .card-corner {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    line-height: 1;
  }}
  .card-corner.bottom {{ transform: rotate(180deg); align-self: flex-end; }}
  .card-rank {{ font-size: 1.15rem; }}
  .card-suit {{ font-size: 0.95rem; margin-top: 1px; }}
  .card-pip {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.6rem;
    opacity: 0.18;
  }}
  .suit-red {{ color: var(--suit-red); }}
  .suit-black {{ color: var(--suit-black); }}
  .card.board-card {{ width: 50px; height: 72px; }}
  .card.mini {{ width: 24px; height: 34px; padding: 2px; }}
  .card.mini .card-rank {{ font-size: 0.6rem; }}
  .card.mini .card-suit {{ font-size: 0.5rem; }}
  .card.mini .card-pip {{ display: none; }}

  .card-back {{
    background:
      repeating-linear-gradient(45deg, #5a2929 0 4px, #4a1f1f 4px 8px),
      #4a1f1f;
    border: 1px solid #2c1010;
  }}
  .card-back-pattern {{
    position: absolute;
    inset: 4px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
  }}

  /* ─── Brain pane ─── */
  .brain-header {{
    border-bottom: 1px solid var(--rule);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
  }}
  .brain-header .role {{
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.5);
  }}
  .brain-header h2 {{
    font-family: var(--serif);
    font-weight: 500;
    font-style: italic;
    font-size: 2.1rem;
    margin: 0.2rem 0 0.4rem;
    color: var(--paper);
  }}
  .brain-header .subtitle {{
    font-family: var(--serif-body);
    font-size: 0.95rem;
    font-style: italic;
    color: rgba(245, 239, 226, 0.6);
  }}

  .module {{
    border-top: 1px solid var(--rule);
    padding: 1.25rem 0 1rem;
  }}
  .module:first-of-type {{ border-top: none; padding-top: 0; }}
  .module-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.7rem;
  }}
  .module-title {{
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 700;
  }}

  /* ─── Number displays ─── */
  .stat-row {{
    display: grid;
    grid-template-columns: 8rem 1fr auto;
    gap: 0.8rem;
    align-items: center;
    margin: 0.4rem 0;
    font-family: var(--mono);
    font-size: 0.85rem;
  }}
  .stat-row .label {{
    color: rgba(245, 239, 226, 0.6);
    font-size: 0.75rem;
  }}
  .stat-row .number {{
    text-align: right;
    color: var(--paper);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }}
  .number.positive {{ color: var(--jade); }}
  .number.negative {{ color: var(--sienna); }}
  .number.highlight {{ color: var(--gold); font-weight: 700; }}

  .bar {{
    height: 6px;
    background: rgba(245, 239, 226, 0.08);
    border-radius: 1px;
    overflow: hidden;
    position: relative;
  }}
  .bar-fill {{
    height: 100%;
    background: var(--paper);
    transition: width 1.2s cubic-bezier(0.2, 0.8, 0.3, 1);
  }}
  .bar-fill.gold {{ background: var(--gold); }}
  .bar-fill.jade {{ background: var(--jade); }}
  .bar-fill.required {{ background: rgba(183, 104, 71, 0.5); }}

  .figures {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin: 0.8rem 0 0.4rem;
  }}
  .figure {{
    border-left: 2px solid var(--gold);
    padding: 0.2rem 0.6rem;
  }}
  .figure .v {{
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.45rem;
    line-height: 1.1;
    color: var(--paper);
  }}
  .figure .l {{
    display: block;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.55);
    margin-top: 2px;
  }}

  /* ─── Explainer dropdowns ─── */
  details.explainer {{
    margin-top: 0.8rem;
    border-top: 1px dashed var(--rule);
    padding-top: 0.6rem;
  }}
  details.explainer[open] summary .explainer-icon {{ background: var(--gold); color: var(--ink); }}
  details.explainer summary {{
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.55);
    user-select: none;
  }}
  details.explainer summary::-webkit-details-marker {{ display: none; }}
  details.explainer summary:hover {{ color: var(--paper); }}
  details.explainer summary:hover .explainer-icon {{ border-color: var(--paper); color: var(--paper); }}
  .explainer-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 1px solid rgba(245, 239, 226, 0.4);
    font-family: var(--serif);
    font-style: italic;
    font-size: 0.85rem;
    transition: all 0.2s;
  }}
  .explainer-body {{
    margin-top: 0.7rem;
    padding: 0.8rem 1rem 0.9rem;
    background: rgba(245, 239, 226, 0.03);
    border-left: 2px solid var(--gold);
    font-family: var(--serif-body);
    font-size: 1rem;
    line-height: 1.55;
    color: rgba(245, 239, 226, 0.85);
  }}
  .explainer-body em {{ color: var(--gold); font-style: italic; }}
  .explainer-body p {{ margin: 0 0 0.6rem; }}
  .explainer-body p:last-child {{ margin-bottom: 0; }}

  /* ─── Decision panel (the highlight box) ─── */
  .decision {{
    margin-top: 1.5rem;
    padding: 1.5rem;
    background:
      linear-gradient(135deg, rgba(199, 154, 74, 0.10), rgba(199, 154, 74, 0.02));
    border: 1px solid rgba(199, 154, 74, 0.35);
    border-radius: 2px;
    position: relative;
  }}
  .decision::before {{
    content: 'DECISION';
    position: absolute;
    top: -8px; left: 1rem;
    background: var(--ink);
    padding: 0 0.5rem;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    color: var(--gold);
  }}
  .decision-action {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 2.2rem;
    color: var(--paper);
    margin: 0 0 0.6rem;
    line-height: 1.1;
  }}
  .decision-action .verdict {{
    font-style: italic;
    color: var(--jade);
    font-size: 1.1rem;
    font-weight: 400;
    margin-left: 0.6rem;
  }}
  .decision-reason {{
    font-family: var(--serif-body);
    font-size: 1rem;
    line-height: 1.6;
    color: rgba(245, 239, 226, 0.85);
    font-style: italic;
  }}

  /* ─── Pull quote / footer ─── */
  .footer-note {{
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--rule);
    font-family: var(--serif-body);
    font-style: italic;
    font-size: 0.95rem;
    color: rgba(245, 239, 226, 0.5);
    text-align: center;
  }}

  /* ─── Range narrowing visual ─── */
  .range-narrow {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.4rem 0;
  }}
  .range-block {{
    text-align: center;
    flex: 1;
  }}
  .range-block .pct {{
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.6rem;
    color: var(--paper);
  }}
  .range-block .lbl {{
    display: block;
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(245, 239, 226, 0.5);
    margin-top: 2px;
  }}
  .range-arrow {{
    font-family: var(--serif);
    font-size: 1.5rem;
    color: var(--gold);
    font-style: italic;
  }}

  /* ─── Stagger entrance ─── */
  .module {{ opacity: 0; transform: translateY(10px); animation: rise 0.6s forwards; }}
  .module:nth-child(1) {{ animation-delay: 0.05s; }}
  .module:nth-child(2) {{ animation-delay: 0.15s; }}
  .module:nth-child(3) {{ animation-delay: 0.25s; }}
  .module:nth-child(4) {{ animation-delay: 0.35s; }}
  .module:nth-child(5) {{ animation-delay: 0.45s; }}
  .module:nth-child(6) {{ animation-delay: 0.55s; }}
  .module:nth-child(7) {{ animation-delay: 0.65s; }}
  .decision {{ opacity: 0; transform: translateY(15px); animation: rise 0.7s 0.8s forwards; }}
  @keyframes rise {{ to {{ opacity: 1; transform: translateY(0); }} }}

  .table-frame {{ opacity: 0; animation: fade-in 0.8s 0.1s forwards; }}
  @keyframes fade-in {{ to {{ opacity: 1; }} }}

</style>
</head>
<body>

<header>
  <div class="brand">poker<span class="accent">·</span><em>training</em></div>
  <div class="meta">Bot's-eye view · TAG_1 · 6-max NLHE · Hand 47</div>
</header>

<div class="grid">

  <!-- ─── LEFT: Table view ─── -->
  <section class="table-pane">
    <div class="table-frame">
      <div class="seat seat-utg folded"><div class="seat-cards">{opponent_cards}</div>UTG·NIT<span class="seat-name">folded</span></div>
      <div class="seat seat-mp folded"><div class="seat-cards">{opponent_cards}</div>MP·MANIAC<span class="seat-name">folded</span></div>
      <div class="seat seat-co"><div class="seat-cards">{opponent_cards}</div>CO·HERO<span class="stack">$97.50</span></div>
      <div class="seat seat-btn folded"><div class="seat-cards">{opponent_cards}</div>BTN·LAG<span class="seat-name">folded</span></div>
      <div class="seat seat-sb folded"><div class="seat-cards">{opponent_cards}</div>SB·STATION<span class="seat-name">folded</span></div>
      <div class="seat seat-bb acting"><div class="seat-cards">{opponent_cards}</div>BB·TAG<span class="stack">$97.00</span></div>

      <div class="pot-display">POT<span class="amount">${POT:.2f}</span></div>
      <div class="board">{board_cards_html}</div>
    </div>

    <div class="hero-area">
      <div class="label">Hero · CO · A♠ K♠</div>
      <div class="hero-cards">{hero_cards_html}</div>
      <div class="hero-info">
        <span>action: <strong style="color:var(--gold)">on you</strong></span>
        <span class="stack">stack: ${HERO_STACK:.2f}</span>
      </div>
    </div>
  </section>

  <!-- ─── RIGHT: The bot's brain ─── -->
  <section class="brain-pane">

    <div class="brain-header">
      <div class="role">DECISION ENGINE · TAG_1</div>
      <h2>What I'm thinking</h2>
      <div class="subtitle">Flop, facing a check from the BB defender · all factors weighed below</div>
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">My Hand</span></div>
      <div class="stat-row"><span class="label">Hole cards</span><span></span><span class="number highlight">A♠ K♠</span></div>
      <div class="stat-row"><span class="label">Hand class</span><span></span><span class="number">AKs (top 1.6%)</span></div>
      <div class="stat-row"><span class="label">Made hand</span><span></span><span class="number positive">Top Pair, Top Kicker</span></div>
      {explainer("hand class", "<p>Each starting hand belongs to one of <em>169 distinct classes</em> — 13 pocket pairs, 78 suited combinations, 78 offsuit. AKs is among the strongest non-pair holdings, ranking in the top 1.6% by preflop equity.</p><p>Knowing your class helps you compare situations across hands without obsessing over specific suits.</p>", "hand")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Equity</span></div>
      <div class="stat-row">
        <span class="label">vs Random</span>
        <div class="bar"><div class="bar-fill" style="width: {eq_random_bar_width}%"></div></div>
        <span class="number">{s.eq_vs_random:.1f}%</span>
      </div>
      <div class="stat-row">
        <span class="label">vs Villain Range</span>
        <div class="bar"><div class="bar-fill jade" style="width: {eq_range_bar_width}%"></div></div>
        <span class="number positive">{s.eq_vs_range:.1f}%</span>
      </div>
      {explainer("equity", f"<p><em>Equity</em> is your probability of winning the hand at showdown if all remaining cards are dealt randomly. We compute it by Monte Carlo: deal thousands of possible runouts, count wins and ties.</p><p>The two numbers reveal an important nuance — your <em>equity vs random hands</em> ({s.eq_vs_random:.0f}%) is generously high. But your <em>equity vs the actual range villain is plausibly holding</em> ({s.eq_vs_range:.0f}%) is what matters for the decision. The gap is the difference between thinking villain has 'anything' versus 'something they'd play preflop and call a raise with.'</p>", "equity")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Pot Odds &amp; EV</span></div>
      <div class="figures">
        <div class="figure"><span class="v">${POT:.0f}</span><span class="l">Current Pot</span></div>
        <div class="figure"><span class="v">${s.bot_bet_size:.2f}</span><span class="l">My Bet Plan</span></div>
        <div class="figure"><span class="v">{s.pot_odds_required_pct:.0f}%</span><span class="l">Villain Needs</span></div>
      </div>
      <div class="stat-row"><span class="label">Required eq</span><span></span><span class="number">{s.pot_odds_required_pct:.1f}%</span></div>
      <div class="stat-row"><span class="label">Actual eq</span><span></span><span class="number positive">{s.eq_vs_range:.1f}%</span></div>
      <div class="stat-row"><span class="label">Edge</span><span></span><span class="number positive">+{s.eq_vs_range - s.pot_odds_required_pct:.1f}pp</span></div>
      {explainer("pot odds", "<p><em>Pot odds</em> are the price the pot is offering. If villain has to call $2 into a $10 pot, they're getting 5-to-1 odds and need only 17% equity to break even on the call.</p><p>The reverse is true for us: when we bet, we're giving villain a price. A small bet gives them better odds (they'll call wider); a big bet gives them worse odds (they fold more, but pay more when they catch). The optimal sizing depends on board texture and villain's range.</p><p><em>Edge</em> is your equity advantage over the break-even threshold — bigger edge means the spot is more profitable.</p>", "potodds")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Opponent Read · BB</span></div>
      <div class="stat-row"><span class="label">Hands seen</span><span></span><span class="number">{VILLAIN_STATS.hands_played}</span></div>
      <div class="stat-row"><span class="label">VPIP</span><span></span><span class="number">{VILLAIN_STATS.vpip * 100:.0f}%</span></div>
      <div class="stat-row"><span class="label">PFR</span><span></span><span class="number">{VILLAIN_STATS.pfr * 100:.0f}%</span></div>
      <div class="stat-row"><span class="label">Fold to cbet</span><span></span><span class="number highlight">{VILLAIN_STATS.fold_to_cbet * 100:.0f}%</span></div>
      <div class="stat-row"><span class="label">Aggression</span><span></span><span class="number">{VILLAIN_STATS.aggression_factor:.1f}</span></div>
      <div class="stat-row"><span class="label">Classified as</span><span></span><span class="number highlight">{s.villain_archetype}</span></div>
      {explainer("opponent stats", "<p>Four numbers tell most of the story:</p><p><em>VPIP</em> (Voluntarily Put In Pot) — how often they enter pots. Low (<15%) means they only play premium hands; high (>40%) means they play almost anything.</p><p><em>PFR</em> (Pre-Flop Raise) — how often they raise preflop. The gap between VPIP and PFR shows how passive they are — a player with VPIP 30% but PFR 5% is calling a lot but rarely raising (a calling station).</p><p><em>Fold to C-bet</em> — when they call your preflop raise and miss the flop, how often do they fold to your continuation bet? This is your <em>fold equity</em> for c-bet bluffs. 47% means almost half your bluffs work outright.</p><p><em>Aggression Factor</em> — bets+raises divided by calls. AF > 2 is aggressive, AF < 1 is passive.</p>", "stats")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Inferred Range</span></div>
      <div class="range-narrow">
        <div class="range-block"><span class="pct">{villain_range_pct_initial:.0f}%</span><span class="lbl">TAG opening</span></div>
        <div class="range-arrow">→</div>
        <div class="range-block"><span class="pct">{villain_range_pct_now:.0f}%</span><span class="lbl">After call</span></div>
      </div>
      <div class="stat-row"><span class="label">Combos now</span><span></span><span class="number">{s.villain_range_combos_now:.0f} of 1326</span></div>
      {explainer("range narrowing", "<p>Each action a player takes <em>narrows</em> what hands they could plausibly hold. A TAG opens about 18% of starting hands. When they call your preflop raise (rather than 3-bet), they remove the very top of their range (which would 3-bet) and the bottom (which would fold) — leaving roughly the middle.</p><p>This narrowing is why <em>equity vs range</em> matters more than equity vs random. By the river, after multiple actions, villain's range can be narrowed to a handful of hands — sometimes you can reason about their exact holdings.</p><p>The combos counter shows the actual count: of 1326 possible 2-card combinations, how many are still in their range.</p>", "ranges")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Board Texture</span></div>
      <div class="stat-row"><span class="label">Surface</span><span></span><span class="number highlight">{s.texture_label}</span></div>
      <div class="stat-row"><span class="label">My sizing</span><span></span><span class="number">${s.bot_bet_size:.2f} ({s.bot_bet_size / POT:.2f}× pot)</span></div>
      {explainer("board texture", "<p>Boards have <em>texture</em> — patterns of suits, connectedness, and pairing that change which hands are likely to be strong.</p><p><em>Dry boards</em> (rainbow, disconnected, no pair like K♥ 7♣ 2♦) reward small probing bets — there are few draws to charge, and your strong hands extract thin value most efficiently with smaller sizing.</p><p><em>Wet boards</em> (monotone, connected, paired like 8♠ 7♠ 5♣) deserve bigger bets to charge the many draws available. If villain has a flush draw, you want them paying full price to chase.</p>", "texture")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Stack-to-Pot</span></div>
      <div class="stat-row"><span class="label">Effective stack</span><span></span><span class="number">${min(HERO_STACK, VILLAIN_STACK):.2f}</span></div>
      <div class="stat-row"><span class="label">Pot</span><span></span><span class="number">${POT:.2f}</span></div>
      <div class="stat-row"><span class="label">SPR</span><span></span><span class="number highlight">{s.spr:.1f} <span style="color:rgba(245,239,226,0.4)">· deep</span></span></div>
      {explainer("SPR", "<p><em>Stack-to-Pot Ratio</em> is the effective stack (smaller of yours and villain's) divided by the current pot. It tells you how committed you are.</p><p><em>SPR &lt; 2</em> — you're already heavily invested. Get the rest in with anything decent.</p><p><em>SPR 2-6</em> — standard depth. Standard play.</p><p><em>SPR &gt; 6</em> — deep stacks. Be careful committing your stack with marginal hands; one pair on the flop is rarely worth a 100bb pot when stacks are 200bb deep.</p><p>SPR fundamentally changes optimal play. The same hand strength is a snap-call at low SPR and a clear fold at high SPR.</p>", "spr")}
    </div>

    <div class="module">
      <div class="module-header"><span class="module-title">Fold Equity</span></div>
      <div class="stat-row"><span class="label">Estimated fold %</span><span></span><span class="number highlight">{s.fold_equity_estimate:.0f}%</span></div>
      <div class="stat-row"><span class="label">Source</span><span></span><span class="number" style="font-size:0.78rem">observed FvCB ({VILLAIN_STATS.cbets_faced} samples)</span></div>
      {explainer("fold equity", "<p><em>Fold equity</em> is the probability your bet causes villain to fold. It's the other half of EV — pure equity assumes you go to showdown, but most pots end before that.</p><p>EV(bet) = fold_eq × pot + (1 − fold_eq) × (eq × (pot + 2 × bet) − bet). When fold equity is high, even weak hands become profitable bets (bluffs). When fold equity is low (vs a calling station), only strong hands should bet for value — bluffs don't work.</p><p>The trainer estimates fold equity from observed behavior: how often did this villain actually fold to similar bets in past hands?</p>", "foldeq")}
    </div>

    <div class="decision">
      <div class="decision-action">{s.decision} <span class="verdict">+EV</span></div>
      <div class="decision-reason">{s.decision_reason}</div>
    </div>

    <div class="footer-note">
      The bot is heuristic, not optimal. The framework is right; the exact decisions are reasonable approximations.<br>
      Watch the inputs, learn the framework — that's the lesson.
    </div>

  </section>
</div>

</body>
</html>"""


def main() -> None:
    snapshot = compute()
    html = render(snapshot)
    out = Path(__file__).parent.parent / "dashboard" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"Open with: open {out}")


if __name__ == "__main__":
    main()
