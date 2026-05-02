"""Page shell — the full HTML scaffold with CSS, fonts, HTMX setup.

The dynamic content (table panel + brain panel) is injected into a placeholder.
"""

from __future__ import annotations

PAGE_CSS = """
:root {
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
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--ink);
  color: var(--paper);
  font-family: var(--serif-body);
  font-size: 17px;
  line-height: 1.5;
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse at 30% 20%, rgba(111, 145, 115, 0.08), transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(199, 154, 74, 0.04), transparent 50%);
}
header {
  border-bottom: 1px solid var(--rule);
  padding: 1.4rem 2.5rem 1.2rem;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}
.brand { font-family: var(--serif); font-style: italic; font-weight: 500; font-size: 1.3rem; letter-spacing: 0.5px; }
.brand .accent { color: var(--gold); font-style: normal; }
.meta { font-family: var(--mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(245, 239, 226, 0.55); }
.grid {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
  gap: 0;
  min-height: calc(100vh - 70px);
}
.table-pane {
  padding: 2rem 2.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  border-right: 1px solid var(--rule);
  position: relative;
}
.brain-pane {
  padding: 1.5rem 2.5rem 4rem;
  overflow-y: auto;
  max-height: calc(100vh - 70px);
}

/* Table */
.table-frame {
  width: 100%;
  max-width: 600px;
  aspect-ratio: 1.5 / 1;
  background: radial-gradient(ellipse at center, #1f2a1f 0%, #131b13 70%);
  border-radius: 50% / 60%;
  border: 8px solid #4a3a1f;
  box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.5), 0 30px 60px rgba(0, 0, 0, 0.4);
  position: relative;
  margin-bottom: 1.5rem;
}
.seat {
  position: absolute;
  text-align: center;
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: rgba(245, 239, 226, 0.7);
}
.seat-info { display: flex; flex-direction: column; gap: 1px; margin-top: 4px; }
.seat-pos { color: var(--gold); font-size: 0.62rem; }
.seat-id { font-size: 0.7rem; }
.seat-name { display: block; margin-top: 4px; font-size: 0.6rem; opacity: 0.5; font-style: italic; }
.seat .stack { display: block; color: var(--gold); font-weight: 700; font-size: 0.78rem; margin-top: 2px; }
.seat-cards { display: flex; gap: 2px; justify-content: center; }
.bet-this-round {
  margin-top: 4px;
  font-size: 0.7rem;
  color: var(--paper);
  background: rgba(199, 154, 74, 0.2);
  padding: 1px 6px;
  border-radius: 2px;
  display: inline-block;
}

.ring-bottom { bottom: -38px; left: 50%; transform: translateX(-50%); }
.ring-bottom-right { right: -10px; bottom: 12%; }
.ring-right { right: -30px; top: 50%; transform: translateY(-50%); }
.ring-top { top: -38px; left: 50%; transform: translateX(-50%); }
.ring-left { left: -30px; top: 50%; transform: translateY(-50%); }
.ring-bottom-left { left: -10px; bottom: 12%; }

.seat.folded { opacity: 0.35; }
.seat.acting .seat-id { color: var(--gold); }
.seat.acting::after {
  content: '';
  display: block;
  width: 6px; height: 6px;
  background: var(--gold);
  border-radius: 50%;
  margin: 6px auto 0;
  box-shadow: 0 0 12px var(--gold);
  animation: pulse 1.6s ease-in-out infinite;
}
.seat.hero-seat .seat-id { color: var(--paper); }
.seat.hero-seat .seat-pos { color: var(--gold); font-weight: 700; }
@keyframes pulse { 50% { opacity: 0.4; transform: scale(0.8); } }

.pot-display {
  position: absolute; top: 30%; left: 50%; transform: translateX(-50%);
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(245, 239, 226, 0.6); text-align: center;
}
.pot-display .amount {
  display: block; font-family: var(--serif); font-style: italic;
  font-size: 1.8rem; color: var(--gold); margin-top: 4px; letter-spacing: 0;
}
.board {
  position: absolute; top: 50%; left: 50%; transform: translateX(-50%);
  display: flex; gap: 6px;
}
.board-empty {
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(245, 239, 226, 0.3); padding-top: 18px;
}

/* Cards */
.card {
  width: 56px; height: 80px;
  background: var(--paper);
  border-radius: 6px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4), inset 0 0 0 1px rgba(0, 0, 0, 0.05);
  position: relative;
  display: flex; flex-direction: column; justify-content: space-between;
  padding: 4px 6px;
  font-family: 'Iowan Old Style', Georgia, serif;
  font-weight: 600;
  background-image: linear-gradient(135deg, rgba(0,0,0,0.02) 0%, transparent 50%);
}
.card-corner { display: flex; flex-direction: column; align-items: flex-start; line-height: 1; }
.card-corner.bottom { transform: rotate(180deg); align-self: flex-end; }
.card-rank { font-size: 1.15rem; }
.card-suit { font-size: 0.95rem; margin-top: 1px; }
.card-pip {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  font-size: 1.6rem; opacity: 0.18;
}
.suit-red { color: var(--suit-red); }
.suit-black { color: var(--suit-black); }
.card.board-card { width: 50px; height: 72px; }
.card.seat-card { width: 40px; height: 56px; padding: 3px 4px; }
.card.seat-card .card-rank { font-size: 0.85rem; }
.card.seat-card .card-suit { font-size: 0.7rem; }
.card.seat-card .card-pip { font-size: 1.2rem; }
.card.mini { width: 22px; height: 30px; padding: 2px; }
.card.mini .card-rank { font-size: 0.55rem; }
.card.mini .card-suit { font-size: 0.5rem; }
.card.mini .card-pip { display: none; }
.card.showdown-card { width: 48px; height: 68px; }

.card-back {
  background: repeating-linear-gradient(45deg, #5a2929 0 4px, #4a1f1f 4px 8px), #4a1f1f;
  border: 1px solid #2c1010;
}
.card-back-pattern {
  position: absolute; inset: 4px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 3px;
}

/* Brain */
.brain-header { border-bottom: 1px solid var(--rule); padding-bottom: 0.8rem; margin-bottom: 1.2rem; }
.brain-header .role { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--gold); }
.brain-header h2 { font-family: var(--serif); font-weight: 500; font-style: italic; font-size: 2rem; margin: 0.2rem 0 0.4rem; }
.brain-header .subtitle { font-family: var(--serif-body); font-size: 0.92rem; font-style: italic; color: rgba(245, 239, 226, 0.6); }

.module { border-top: 1px solid var(--rule); padding: 1rem 0 0.8rem; }
.module:first-of-type { border-top: none; padding-top: 0; }
.module-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.5rem; }
.module-title { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--gold); font-weight: 700; }

.stat-row {
  display: grid; grid-template-columns: 8rem 1fr auto;
  gap: 0.8rem; align-items: center; margin: 0.35rem 0;
  font-family: var(--mono); font-size: 0.83rem;
}
.stat-row .label { color: rgba(245, 239, 226, 0.6); font-size: 0.74rem; }
.stat-row .number { text-align: right; color: var(--paper); font-weight: 500; font-variant-numeric: tabular-nums; }
.number.positive { color: var(--jade); }
.number.negative { color: var(--sienna); }
.number.highlight { color: var(--gold); font-weight: 700; }

.bar { height: 6px; background: rgba(245, 239, 226, 0.08); border-radius: 1px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--paper); transition: width 0.8s cubic-bezier(0.2, 0.8, 0.3, 1); }
.bar-fill.gold { background: var(--gold); }
.bar-fill.jade { background: var(--jade); }

.figures { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin: 0.7rem 0 0.4rem; }
.figure { border-left: 2px solid var(--gold); padding: 0.2rem 0.6rem; }
.figure .v { font-family: var(--serif); font-style: italic; font-size: 1.4rem; line-height: 1.1; color: var(--paper); }
.figure .l { display: block; font-family: var(--mono); font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(245, 239, 226, 0.55); margin-top: 2px; }

/* Explainers */
details.explainer { margin-top: 0.6rem; border-top: 1px dashed var(--rule); padding-top: 0.5rem; }
details.explainer[open] summary .explainer-icon { background: var(--gold); color: var(--ink); }
details.explainer summary {
  list-style: none; cursor: pointer;
  display: flex; align-items: center; gap: 0.6rem;
  font-family: var(--mono); font-size: 0.68rem; letter-spacing: 1.2px;
  text-transform: uppercase; color: rgba(245, 239, 226, 0.55); user-select: none;
}
details.explainer summary::-webkit-details-marker { display: none; }
details.explainer summary:hover { color: var(--paper); }
details.explainer summary:hover .explainer-icon { border-color: var(--paper); color: var(--paper); }
.explainer-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 50%;
  border: 1px solid rgba(245, 239, 226, 0.4);
  font-family: var(--serif); font-style: italic; font-size: 0.85rem;
  transition: all 0.2s;
}
.explainer-body {
  margin-top: 0.6rem; padding: 0.7rem 0.9rem 0.8rem;
  background: rgba(245, 239, 226, 0.03);
  border-left: 2px solid var(--gold);
  font-family: var(--serif-body); font-size: 0.97rem; line-height: 1.55;
  color: rgba(245, 239, 226, 0.85);
}
.explainer-body em { color: var(--gold); font-style: italic; }
.explainer-body p { margin: 0 0 0.5rem; }
.explainer-body p:last-child { margin-bottom: 0; }

/* Action area */
.action-area {
  margin-top: 1.5rem; padding-top: 1rem;
  border-top: 2px solid var(--gold);
}
.action-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.6rem; align-items: center; }
.action-form, .bet-form { display: inline-flex; gap: 0.4rem; align-items: center; }
.action-btn {
  padding: 0.6rem 1.1rem;
  background: rgba(245, 239, 226, 0.05);
  color: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 2px;
  font-family: var(--mono); font-size: 0.78rem;
  letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;
  cursor: pointer; transition: all 0.15s;
}
.action-btn:hover { background: rgba(245, 239, 226, 0.1); border-color: var(--gold); }
.action-btn.primary { background: var(--gold); color: var(--ink); border-color: var(--gold); }
.action-btn.primary:hover { background: #d8af5e; }
.action-btn.negative { color: var(--sienna); border-color: rgba(183, 104, 71, 0.5); }
.action-btn.negative:hover { background: rgba(183, 104, 71, 0.1); }
.action-btn.quick { padding: 0.4rem 0.7rem; font-size: 0.7rem; opacity: 0.8; }
.bet-input {
  background: var(--ink); color: var(--paper);
  border: 1px solid var(--rule); border-radius: 2px;
  padding: 0.55rem 0.7rem;
  font-family: var(--mono); font-size: 0.95rem; font-weight: 700;
  width: 100px; text-align: right;
}
.bet-input:focus { outline: none; border-color: var(--gold); }

.next-hand-cta { padding: 2rem 0; text-align: center; }
.next-hand-cta .action-btn { padding: 1rem 2rem; font-size: 0.95rem; }

/* Action log */
.action-log {
  margin-top: 1rem; max-width: 600px; width: 100%;
  border: 1px solid var(--rule); border-radius: 2px;
  padding: 0.5rem 0.75rem; max-height: 220px; overflow-y: auto;
}
.log-row {
  display: grid; grid-template-columns: 3rem 6rem 1fr auto;
  gap: 0.6rem; padding: 0.2rem 0;
  font-family: var(--mono); font-size: 0.75rem;
  border-bottom: 1px dashed rgba(245, 239, 226, 0.05);
}
.log-row:last-child { border-bottom: none; }
.log-row.you { color: var(--gold); font-weight: 700; }
.log-street { color: rgba(245, 239, 226, 0.4); }
.log-id { color: rgba(245, 239, 226, 0.7); }
.log-action { text-transform: uppercase; }
.log-amt { color: var(--paper); text-align: right; }
.log-empty { font-family: var(--mono); font-size: 0.8rem; color: rgba(245, 239, 226, 0.3); padding: 0.5rem; text-align: center; }

/* Showdown */
.showdown {
  margin-top: 1rem; padding: 1rem 1.2rem;
  background: linear-gradient(135deg, rgba(199, 154, 74, 0.10), rgba(199, 154, 74, 0.02));
  border: 1px solid rgba(199, 154, 74, 0.35); border-radius: 2px;
  max-width: 600px; width: 100%;
}
.showdown-title {
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 2.5px;
  color: var(--gold); margin-bottom: 0.6rem;
}
.showdown-result { font-family: var(--serif); font-style: italic; font-size: 1.1rem; }
.showdown-players { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.7rem; }
.showdown-player {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 0.4rem 0.6rem; border-radius: 2px;
}
.showdown-player.won { background: rgba(111, 145, 115, 0.15); border: 1px solid rgba(111, 145, 115, 0.4); }
.showdown-name { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 1px; color: rgba(245, 239, 226, 0.7); }
.showdown-cards { display: flex; gap: 4px; }
.won-amt { font-family: var(--mono); font-size: 0.7rem; color: var(--jade); font-weight: 700; }
.pot-line { font-family: var(--mono); font-size: 0.78rem; color: rgba(245, 239, 226, 0.7); padding: 2px 0; }

/* Thinking spinner */
.thinking-spinner {
  display: flex; gap: 8px; margin: 2rem auto; justify-content: center;
}
.thinking-spinner div {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--gold); opacity: 0.4;
  animation: thinking 1.2s ease-in-out infinite;
}
.thinking-spinner div:nth-child(2) { animation-delay: 0.15s; }
.thinking-spinner div:nth-child(3) { animation-delay: 0.3s; }
@keyframes thinking { 50% { opacity: 1; transform: scale(1.3); } }

/* Smooth fade-in for HTMX swaps */
#main { animation: fade-in 0.3s; }
@keyframes fade-in { from { opacity: 0.3; } to { opacity: 1; } }
"""


def render_page(table_html: str, brain_html: str, log_html: str, showdown_html: str = "",
                hand_id: int = 0, button_id: str = "?") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Poker Training — Live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<style>{PAGE_CSS}</style>
</head>
<body>

<header>
  <div class="brand">poker<span class="accent">·</span><em>training</em></div>
  <div class="meta">Live · 6-max NLHE · Hand #{hand_id} · Button: {button_id}</div>
</header>

<div id="main">
<div class="grid">
  <section class="table-pane">
    {table_html}
    {showdown_html}
    {log_html}
  </section>
  <section class="brain-pane">
    {brain_html}
  </section>
</div>
</div>

</body>
</html>"""
