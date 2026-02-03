"""Fair Value Gap (FVG) + inversion (iFVG).

Definitions (ICT-style 3-candle gap):
- Bullish FVG: candle1.high < candle3.low
- Bearish FVG: candle1.low  > candle3.high

We track zones as (low, high) price ranges and watch for price to trade back into
that zone. An "inversion" is a simple confirmation that price reacted in the
reversal direction after re-entering the zone.

This module is deliberately conservative/simplified for a first pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class FVG:
    fvg_type: str  # "BULL" or "BEAR"
    low: float
    high: float
    created_at: datetime
    timeframe_min: int

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


def detect_fvgs(
    df: pd.DataFrame,
    timeframe_min: int,
    min_gap_points: float = 0.25,
    max_age_min: int = 120,
) -> List[FVG]:
    if df is None or df.empty or len(df) < 3:
        return []

    now = df.index[-1].to_pydatetime()
    max_age = timedelta(minutes=int(max_age_min))

    out: List[FVG] = []

    # iterate over 3-candle windows
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]
        t3 = df.index[i].to_pydatetime()

        # Bullish gap between c1.high and c3.low
        gap_low = float(c1["high"])
        gap_high = float(c3["low"])
        if gap_high - gap_low >= float(min_gap_points):
            if now - t3 <= max_age:
                out.append(
                    FVG(
                        fvg_type="BULL",
                        low=gap_low,
                        high=gap_high,
                        created_at=t3,
                        timeframe_min=int(timeframe_min),
                    )
                )

        # Bearish gap between c3.high and c1.low (inverted ordering)
        gap_low2 = float(c3["high"])
        gap_high2 = float(c1["low"])
        if gap_high2 - gap_low2 >= float(min_gap_points):
            if now - t3 <= max_age:
                out.append(
                    FVG(
                        fvg_type="BEAR",
                        low=gap_low2,
                        high=gap_high2,
                        created_at=t3,
                        timeframe_min=int(timeframe_min),
                    )
                )

    # De-dup near-identical zones (same type+tf, small tolerance)
    out_sorted = sorted(out, key=lambda z: (z.timeframe_min, z.fvg_type, z.created_at))
    dedup: List[FVG] = []
    tol = 0.01
    for z in out_sorted:
        if not dedup:
            dedup.append(z)
            continue
        prev = dedup[-1]
        if (
            prev.fvg_type == z.fvg_type
            and prev.timeframe_min == z.timeframe_min
            and abs(prev.low - z.low) <= tol
            and abs(prev.high - z.high) <= tol
        ):
            continue
        dedup.append(z)

    return dedup


def detect_ifvg(
    *,
    df: pd.DataFrame,
    fvg: FVG,
    reaction_candles: int = 2,
) -> Optional[str]:
    """Return entry side ("LONG" or "SHORT") if an iFVG reaction is confirmed.

    - Bullish iFVG: price returns into a BEAR FVG zone and then closes up (bullish reaction)
    - Bearish iFVG: price returns into a BULL FVG zone and then closes down (bearish reaction)

    Confirmation logic:
    - One of the last `reaction_candles` candles must trade into the zone.
    - The most recent candle close must be in the reversal direction.
    """
    if df is None or df.empty:
        return None

    n = max(3, int(reaction_candles))
    tail = df.tail(n)
    if tail.empty:
        return None

    last = tail.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])

    traded_into = False
    for _, row in tail.iterrows():
        if float(row["low"]) <= fvg.high and float(row["high"]) >= fvg.low:
            traded_into = True
            break

    if not traded_into:
        return None

    # Reaction direction from the last candle body
    bullish = last_close > last_open
    bearish = last_close < last_open

    if fvg.fvg_type == "BEAR" and bullish:
        return "LONG"
    if fvg.fvg_type == "BULL" and bearish:
        return "SHORT"

    return None
