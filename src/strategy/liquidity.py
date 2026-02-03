"""Liquidity levels + sweep detection.

This is intentionally simple:
- Identify recent swing highs/lows (pivot extremes) using a rolling lookback.
- Detect a "sweep" when price trades beyond ORB high/low by a buffer and then
  closes back inside the ORB range (optional).

The goal is to produce reliable *events* for the state machine, not perfect market structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SweepEvent:
    side: str  # "BUY" (sweep highs) or "SELL" (sweep lows)
    sweep_price: float
    at: datetime


def recent_swings(df: pd.DataFrame, lookback: int = 20) -> tuple[Optional[float], Optional[float]]:
    """Return (swing_high, swing_low) from the last `lookback` candles."""
    if df is None or df.empty:
        return None, None

    tail = df.tail(int(lookback))
    if tail.empty:
        return None, None

    return float(tail["high"].max()), float(tail["low"].min())


def detect_sweep(
    *,
    df_1m: pd.DataFrame,
    orb_high: float,
    orb_low: float,
    buffer_points: float = 0.5,
    require_close_back_inside: bool = True,
) -> Optional[SweepEvent]:
    if df_1m is None or df_1m.empty:
        return None

    last = df_1m.iloc[-1]
    ts = df_1m.index[-1].to_pydatetime()

    hi = float(last["high"])
    lo = float(last["low"])
    close = float(last["close"])

    swept_high = hi >= (orb_high + buffer_points)
    swept_low = lo <= (orb_low - buffer_points)

    if require_close_back_inside:
        back_inside = orb_low <= close <= orb_high
    else:
        back_inside = True

    if swept_high and back_inside:
        return SweepEvent(side="BUY", sweep_price=hi, at=ts)
    if swept_low and back_inside:
        return SweepEvent(side="SELL", sweep_price=lo, at=ts)

    return None
