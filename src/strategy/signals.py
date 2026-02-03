"""Signal composition for ORB + sweep + SMT + (i)FVG.

This module converts detected events into a paper-trade "entry plan".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EntryPlan:
    side: str  # LONG|SHORT
    entry: float
    stop: float
    target: float
    reason: str
    timeframe_min: int


def build_entry_plan(
    *,
    side: str,
    entry: float,
    sweep_price: float,
    stop_buffer: float,
    rr: float,
    timeframe_min: int,
    orb_high: float,
    orb_low: float,
) -> EntryPlan:
    side_u = side.upper()

    if side_u == "LONG":
        stop = sweep_price - float(stop_buffer)
        risk = max(0.25, entry - stop)
        target = entry + risk * float(rr)
        reason = f"iFVG LONG after sell-side sweep; ORB={orb_low:.2f}-{orb_high:.2f}"
    else:
        stop = sweep_price + float(stop_buffer)
        risk = max(0.25, stop - entry)
        target = entry - risk * float(rr)
        reason = f"iFVG SHORT after buy-side sweep; ORB={orb_low:.2f}-{orb_high:.2f}"

    return EntryPlan(
        side=side_u,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        reason=reason,
        timeframe_min=int(timeframe_min),
    )
