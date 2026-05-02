"""Pot odds, required equity, and EV — the bridge from equity to decision.

Standard formulas (heads-up, single-street perspective):

  required_equity = call / (pot + call)
      The minimum win rate needed for a call to break even.
      Example: pot=80, call=20 → 20/100 = 20% needed.

  EV(call) = equity * pot - (1 - equity) * call
      Expected dollar profit from calling.
      Positive when equity > required_equity. Magnitude = how good the call is.

A call is +EV when actual_equity > required_equity. The DELTA between the two
(in pp or in dollars) tells you how good or bad the spot is, not just whether
to call.

`pot` here means the pot AFTER villain's bet — i.e., what's there to win
before you put in your call. `call` is what you have to add to continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd


@dataclass(frozen=True)
class PotOddsAnalysis:
    pot_size: float
    call_amount: float
    required_equity_pct: float   # 0-100
    actual_equity_pct: float     # 0-100
    pot_odds_ratio: str          # e.g. "4:1"
    ev: float                    # expected $ profit from calling
    verdict: str                 # "+EV call", "-EV call", "Break-even"
    edge_pp: float               # actual - required (percentage points)

    def __str__(self) -> str:
        sign = "+" if self.ev >= 0 else ""
        return (
            f"pot ${self.pot_size:.2f}, call ${self.call_amount:.2f} "
            f"({self.pot_odds_ratio} → need {self.required_equity_pct:.1f}%) | "
            f"have {self.actual_equity_pct:.1f}% | "
            f"edge {self.edge_pp:+.1f}pp | "
            f"EV {sign}${self.ev:.2f} → {self.verdict}"
        )


def required_equity(pot: float, call: float) -> float:
    """Minimum equity needed for a call to break even, as a fraction (0-1)."""
    if call < 0 or pot < 0:
        raise ValueError("pot and call must be non-negative")
    if call == 0:
        return 0.0  # checking is always free
    return call / (pot + call)


def pot_odds_ratio(pot: float, call: float) -> str:
    """Format pot odds as 'X:1' (or 'X:Y' for cleaner small ratios).

    Example: pot=80, call=20 → "4:1". pot=100, call=33.33 → "3:1".
    """
    if call == 0:
        return "free"
    ratio = pot / call
    # Try to express as integer X:1 if close
    if abs(ratio - round(ratio)) < 0.05:
        return f"{round(ratio)}:1"
    # Otherwise use a fractional representation rounded to 1 decimal
    return f"{ratio:.1f}:1"


def ev(equity: float, pot: float, call: float) -> float:
    """Expected value of calling, in dollars. Equity is a fraction 0-1."""
    if not 0 <= equity <= 1:
        raise ValueError(f"equity must be 0-1, got {equity}")
    return equity * pot - (1 - equity) * call


def analyze_call(pot: float, call: float, equity_pct: float) -> PotOddsAnalysis:
    """Full analysis: pot odds, required equity, EV, and verdict.

    `equity_pct` is hero's equity as a percent (0-100), as returned by the
    Equity calculator. This function does the unit conversion.
    """
    if not 0 <= equity_pct <= 100:
        raise ValueError(f"equity_pct must be 0-100, got {equity_pct}")

    eq_frac = equity_pct / 100.0
    req_frac = required_equity(pot, call)
    expected = ev(eq_frac, pot, call)
    edge = equity_pct - req_frac * 100

    # Verdict bands: ±0.5pp tolerance for "break-even"
    if edge > 0.5:
        verdict = "+EV call"
    elif edge < -0.5:
        verdict = "-EV call"
    else:
        verdict = "Break-even"

    return PotOddsAnalysis(
        pot_size=pot,
        call_amount=call,
        required_equity_pct=req_frac * 100,
        actual_equity_pct=equity_pct,
        pot_odds_ratio=pot_odds_ratio(pot, call),
        ev=expected,
        verdict=verdict,
        edge_pp=edge,
    )
