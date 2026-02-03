"""Opening Range (ORB) tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OpeningRange:
    start: datetime
    end: datetime
    high: float
    low: float


class ORBTracker:
    def __init__(self):
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        self._high: Optional[float] = None
        self._low: Optional[float] = None
        self._finalized: bool = False

    def start(self, start_dt: datetime, end_dt: datetime) -> None:
        self._start = start_dt
        self._end = end_dt
        self._high = None
        self._low = None
        self._finalized = False

    def update(self, price_high: float, price_low: float) -> None:
        if self._finalized:
            return
        self._high = price_high if self._high is None else max(self._high, price_high)
        self._low = price_low if self._low is None else min(self._low, price_low)

    def finalize(self) -> OpeningRange:
        if self._start is None or self._end is None:
            raise RuntimeError("ORBTracker not started")
        if self._high is None or self._low is None:
            raise RuntimeError("ORBTracker has no data")
        self._finalized = True
        return OpeningRange(start=self._start, end=self._end, high=float(self._high), low=float(self._low))

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def high(self) -> Optional[float]:
        return self._high

    @property
    def low(self) -> Optional[float]:
        return self._low
