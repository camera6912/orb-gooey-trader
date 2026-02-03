"""SMT divergence detection between /NQ and /ES.

SMT here is implemented as a simple divergence of extremes:
- Bearish SMT: NQ makes a higher high vs lookback while ES does not (or vice versa)
- Bullish SMT: NQ makes a lower low vs lookback while ES does not (or vice versa)

This provides a directional bias after a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SMTSignal:
    smt: str  # "BULLISH" | "BEARISH"
    leader: str  # which market swept/extended ("NQ" or "ES")


def detect_smt(
    *,
    nq: pd.DataFrame,
    es: pd.DataFrame,
    lookback: int = 20,
) -> Optional[SMTSignal]:
    if nq is None or es is None or nq.empty or es.empty:
        return None

    nqt = nq.tail(int(lookback))
    est = es.tail(int(lookback))
    if len(nqt) < 5 or len(est) < 5:
        return None

    nq_prev = nqt.iloc[:-1]
    es_prev = est.iloc[:-1]

    nq_last_hi = float(nqt.iloc[-1]["high"])
    nq_last_lo = float(nqt.iloc[-1]["low"])
    es_last_hi = float(est.iloc[-1]["high"])
    es_last_lo = float(est.iloc[-1]["low"])

    nq_prev_hi = float(nq_prev["high"].max())
    nq_prev_lo = float(nq_prev["low"].min())
    es_prev_hi = float(es_prev["high"].max())
    es_prev_lo = float(es_prev["low"].min())

    nq_made_higher_high = nq_last_hi > nq_prev_hi
    es_made_higher_high = es_last_hi > es_prev_hi

    nq_made_lower_low = nq_last_lo < nq_prev_lo
    es_made_lower_low = es_last_lo < es_prev_lo

    # Bearish divergence: one makes HH, the other does not
    if nq_made_higher_high and not es_made_higher_high:
        return SMTSignal(smt="BEARISH", leader="NQ")
    if es_made_higher_high and not nq_made_higher_high:
        return SMTSignal(smt="BEARISH", leader="ES")

    # Bullish divergence: one makes LL, the other does not
    if nq_made_lower_low and not es_made_lower_low:
        return SMTSignal(smt="BULLISH", leader="NQ")
    if es_made_lower_low and not nq_made_lower_low:
        return SMTSignal(smt="BULLISH", leader="ES")

    return None
