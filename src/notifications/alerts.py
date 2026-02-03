"""Human-facing alert strings.

Keep message formatting out of strategy logic so signals remain testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def format_orb_set(*, symbol: str, high: float, low: float, start: datetime, end: datetime) -> str:
    rng = high - low
    return (
        f"🎯 ORB Set — {symbol} ({start:%-I:%M}–{end:%-I:%M %p} ET)\n"
        f"High: {high:.2f} | Low: {low:.2f} | Range: {rng:.2f} pts"
    )


def format_sweep(*, symbol: str, side: str, sweep_price: float, orb_high: float, orb_low: float) -> str:
    return (
        f"💧 Liquidity Sweep — {symbol}\n"
        f"Side: {side.upper()} | Sweep: {sweep_price:.2f}\n"
        f"ORB: {orb_low:.2f}–{orb_high:.2f}"
    )


def format_smt(*, smt: str, nq: str = "/NQ", es: str = "/ES") -> str:
    return f"🧠 SMT Divergence: {smt} ({nq} vs {es})"


def format_fvg(*, timeframe_min: int, fvg_type: str, low: float, high: float, created_at: datetime) -> str:
    return (
        f"🟦 FVG Detected ({timeframe_min}m) — {fvg_type.upper()}\n"
        f"Zone: {low:.2f}–{high:.2f} | {created_at:%-I:%M:%S %p} ET"
    )


def format_ifvg_entry(
    *,
    symbol: str,
    side: str,
    entry: float,
    stop: float,
    target: float,
    timeframe_min: int,
    reason: str,
) -> str:
    emoji = "🟢" if side.upper() == "LONG" else "🔴"
    return (
        f"🚨 iFVG Entry Signal — {symbol}\n"
        f"{emoji} {side.upper()} @ {entry:.2f} (tf={timeframe_min}m)\n"
        f"Stop: {stop:.2f} | Target: {target:.2f}\n"
        f"Reason: {reason}"
    )


@dataclass(frozen=True)
class ExitSummary:
    exit_reason: str
    pnl_points: float
    duration_s: float
    pnl_dollars: Optional[float] = None


def format_exit(*, symbol: str, side: str, entry: float, exit_price: float, summary: ExitSummary) -> str:
    if summary.exit_reason == "target":
        headline = "✅ Target Hit"
    elif summary.exit_reason == "stop":
        headline = "❌ Stop Hit"
    elif summary.exit_reason == "eod":
        headline = "⏰ EOD Exit"
    else:
        headline = "📤 Exit"

    sign = "+" if summary.pnl_points > 0 else ""
    pts = f"{sign}{summary.pnl_points:.2f} pts"

    mins = int(round(summary.duration_s / 60.0))
    dur = f"{mins} min" if mins < 120 else f"{mins/60.0:.1f} hr"

    return (
        f"{headline} — {symbol} ({pts})\n"
        f"{side.upper()} {entry:.2f} → {exit_price:.2f}\n"
        f"Duration: {dur}"
    )
