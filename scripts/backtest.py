#!/usr/bin/env python3
"""Backtest ORB + sweep + iFVG strategy (no breakout trades).

Per day:
- Build ORB (09:30-09:45 ET) on /NQ
- After 09:45, wait for a liquidity sweep beyond ORB
- After sweep, detect FVGs (1m/3m/5m) and wait for inversion (iFVG)
- Enter on iFVG confirmation
- Stop beyond sweep extreme
- Exit on first touch of:
  - 2R target, OR
  - developing session high/low at entry, OR
  - previous day high/low
  Otherwise exit at 16:00 close.

Outputs a console summary and writes full results to logs/backtest_results.json.

Notes:
- This is a *conservative* bar-based backtest (no tick ordering inside bars).
- If stop and target hit in the same 1m bar, we assume STOP first.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, time):
            return obj.isoformat()
        return super().default(obj)
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

# Allow running as a script: `python3 scripts/backtest.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schwab import SchwabClient
from src.strategy.fvg import FVG, detect_fvgs, detect_ifvg
from src.strategy.liquidity import SweepEvent, detect_sweep
from src.strategy.orb import ORBTracker
from src.strategy.signals import EntryPlan, build_entry_plan
from src.strategy.smt import SMTSignal, detect_smt


NQ_DOLLARS_PER_POINT = 20.0


@dataclass
class DayResult:
    day: str
    analyzed: bool
    skipped_reason: Optional[str] = None

    orb_high: Optional[float] = None
    orb_low: Optional[float] = None

    sweep: Optional[Dict[str, Any]] = None
    smt: Optional[Dict[str, Any]] = None

    entry: Optional[Dict[str, Any]] = None
    exit: Optional[Dict[str, Any]] = None

    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    outcome: str = "SKIP"  # WIN|LOSS|BREAKEVEN|SKIP


def _load_settings(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _trading_days(end_day: date, n: int) -> List[date]:
    """Return the most recent n *weekday* dates ending at end_day (inclusive)."""
    out: List[date] = []
    d = end_day
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = d - timedelta(days=1)
    return list(reversed(out))


def _rth_window(d: date) -> tuple[datetime, datetime, datetime, datetime]:
    start = datetime.combine(d, time(9, 30))
    orb_end = datetime.combine(d, time(9, 45))
    end = datetime.combine(d, time(16, 0))
    return start, orb_end, end, datetime.combine(d, time(0, 0))


def _filter_rth(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    start, _, end, _ = _rth_window(day)
    start = pd.Timestamp(start, tz="America/New_York")
    end = pd.Timestamp(end, tz="America/New_York")
    return df.loc[start:end].copy()


def _resample(df_1m: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    if timeframe_min == 1:
        return df_1m
    rule = f"{int(timeframe_min)}min"
    o = df_1m["open"].resample(rule).first()
    h = df_1m["high"].resample(rule).max()
    l = df_1m["low"].resample(rule).min()
    c = df_1m["close"].resample(rule).last()
    v = df_1m["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()


def _prev_day_levels(client: SchwabClient, symbol: str, d: date) -> tuple[Optional[float], Optional[float]]:
    prev = d - timedelta(days=1)
    # skip weekends
    while prev.weekday() >= 5:
        prev = prev - timedelta(days=1)

    start, _, end, _ = _rth_window(prev)
    df = client.get_history_1m(symbol, start, end)
    df = _filter_rth(df, prev)
    if df.empty:
        return None, None
    return float(df["high"].max()), float(df["low"].min())


def _simulate_trade(
    *,
    df_1m: pd.DataFrame,
    entry_plan: EntryPlan,
    entry_ts: pd.Timestamp,
    prev_high: Optional[float],
    prev_low: Optional[float],
    session_hi_at_entry: float,
    session_lo_at_entry: float,
    eod_ts: pd.Timestamp,
) -> Dict[str, Any]:
    side = entry_plan.side
    stop = entry_plan.stop

    # Compute the 2R target from EntryPlan, then add optional fixed levels.
    targets: List[float] = [float(entry_plan.target)]

    if side == "LONG":
        if prev_high is not None and prev_high > entry_plan.entry:
            targets.append(float(prev_high))
        if session_hi_at_entry > entry_plan.entry:
            targets.append(float(session_hi_at_entry))
        # keep only targets above entry
        targets = [t for t in targets if t > entry_plan.entry]
        targets = sorted(set(targets))
    else:
        if prev_low is not None and prev_low < entry_plan.entry:
            targets.append(float(prev_low))
        if session_lo_at_entry < entry_plan.entry:
            targets.append(float(session_lo_at_entry))
        targets = [t for t in targets if t < entry_plan.entry]
        targets = sorted(set(targets), reverse=True)

    # Walk forward bar-by-bar starting at the next candle after entry confirmation.
    after = df_1m.loc[entry_ts:]
    if after.empty:
        return {
            "exit_ts": str(entry_ts),
            "exit_price": float(entry_plan.entry),
            "exit_reason": "NO_BARS_AFTER_ENTRY",
        }

    # First bar includes entry bar; allow exit on that same bar *after* entry.
    # With only OHLC we can't know ordering; we stay conservative:
    # - if stop is within bar range => assume stop first.
    # - else if any target within bar range => assume take profit.
    for ts, row in after.iterrows():
        if ts > eod_ts:
            break

        hi = float(row["high"])
        lo = float(row["low"])
        close = float(row["close"])

        if side == "LONG":
            stop_hit = lo <= stop
            target_hit = next((t for t in targets if hi >= t), None)
            if stop_hit:
                return {"exit_ts": str(ts), "exit_price": float(stop), "exit_reason": "STOP"}
            if target_hit is not None:
                return {"exit_ts": str(ts), "exit_price": float(target_hit), "exit_reason": "TARGET"}
        else:
            stop_hit = hi >= stop
            target_hit = next((t for t in targets if lo <= t), None)
            if stop_hit:
                return {"exit_ts": str(ts), "exit_price": float(stop), "exit_reason": "STOP"}
            if target_hit is not None:
                return {"exit_ts": str(ts), "exit_price": float(target_hit), "exit_reason": "TARGET"}

        if ts == eod_ts:
            return {"exit_ts": str(ts), "exit_price": float(close), "exit_reason": "EOD"}

    # Fallback: exit at last close
    last_ts = after.index[-1]
    last_close = float(after.iloc[-1]["close"])
    return {"exit_ts": str(last_ts), "exit_price": float(last_close), "exit_reason": "EOD_FALLBACK"}


def backtest_day(
    *,
    client: SchwabClient,
    day: date,
    settings: dict,
    nq_symbol: str = "/NQ",
    es_symbol: str = "/ES",
) -> DayResult:
    res = DayResult(day=str(day), analyzed=True)

    tz = settings.get("timezone", "America/New_York")
    session = settings.get("session", {})
    strat = settings.get("strategy", {})

    orb_cfg = strat.get("orb", {})
    liq_cfg = strat.get("liquidity", {})
    fvg_cfg = strat.get("fvg", {})
    risk_cfg = strat.get("risk", {})

    # Optional skip calendar list
    fomc_days = set((settings.get("calendar", {}) or {}).get("fomc_days", []) or [])
    if str(day) in fomc_days:
        res.outcome = "SKIP"
        res.skipped_reason = "FOMC_DAY"
        return res

    start, orb_end, end, _ = _rth_window(day)
    start_ts = pd.Timestamp(start, tz=tz)
    orb_end_ts = pd.Timestamp(orb_end, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)

    # Fetch 1m for both markets
    nq_1m = client.get_history_1m(nq_symbol, start, end)
    es_1m = client.get_history_1m(es_symbol, start, end)
    nq_1m = _filter_rth(nq_1m, day)
    es_1m = _filter_rth(es_1m, day)

    if nq_1m.empty or es_1m.empty:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_DATA"
        return res

    # Previous day levels (optional target)
    prev_high, prev_low = _prev_day_levels(client, nq_symbol, day)

    # Build ORB
    orb = ORBTracker()
    orb.start(start_ts.to_pydatetime(), orb_end_ts.to_pydatetime())

    orb_slice = nq_1m.loc[start_ts:orb_end_ts - pd.Timedelta(minutes=1)]
    if orb_slice.empty:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_ORB_DATA"
        return res

    for _, row in orb_slice.iterrows():
        orb.update(float(row["high"]), float(row["low"]))

    orbr = orb.finalize()
    res.orb_high = float(orbr.high)
    res.orb_low = float(orbr.low)

    min_orb_range = float(orb_cfg.get("min_range_points", 0.0) or 0.0)
    if (res.orb_high - res.orb_low) < min_orb_range:
        res.outcome = "SKIP"
        res.skipped_reason = f"ORB_RANGE_TOO_SMALL(<{min_orb_range})"
        return res

    # After 09:45 scan for sweep -> then iFVG
    swing_lookback = int(liq_cfg.get("swing_lookback", 20) or 20)
    sweep_buffer = float(liq_cfg.get("sweep_buffer_points", 0.5) or 0.5)
    sweep_close_back = bool(liq_cfg.get("sweep_confirm_close_back_inside", True))

    entry_tfs = list(map(int, (fvg_cfg.get("entry_timeframes_min") or [1, 3, 5])))
    min_gap = float(fvg_cfg.get("min_gap_points", 0.25) or 0.25)
    max_age = int(fvg_cfg.get("max_fvg_age_min", 120) or 120)
    reaction_candles = int(fvg_cfg.get("ifvg_reaction_candles", 2) or 2)

    stop_buffer = float(risk_cfg.get("stop_buffer_points", 0.5) or 0.5)
    rr = float(risk_cfg.get("default_rr", 2.0) or 2.0)

    sweep_event: Optional[SweepEvent] = None
    smt_sig: Optional[SMTSignal] = None

    # iterate minute by minute after ORB end
    post = nq_1m.loc[orb_end_ts:end_ts]
    if post.empty:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_POST_ORB_DATA"
        return res

    active_fvgs: List[FVG] = []
    entry_plan: Optional[EntryPlan] = None
    entry_ts: Optional[pd.Timestamp] = None

    for ts, _ in post.iterrows():
        cur_nq = nq_1m.loc[:ts]
        cur_es = es_1m.loc[:ts]

        if sweep_event is None:
            # detect sweep based on the last candle at ts
            sweep_event = detect_sweep(
                df_1m=cur_nq.tail(max(60, swing_lookback)),
                orb_high=res.orb_high,
                orb_low=res.orb_low,
                buffer_points=sweep_buffer,
                require_close_back_inside=sweep_close_back,
            )
            if sweep_event is not None:
                res.sweep = asdict(sweep_event)
                # SMT at sweep moment (for logging / comparison)
                smt_sig = detect_smt(nq=cur_nq.tail(60), es=cur_es.tail(60), lookback=20)
                if smt_sig is not None:
                    res.smt = asdict(smt_sig)
            continue

        # After sweep: keep updating FVG list and check for iFVG
        sweep_ts = pd.Timestamp(sweep_event.at)
        if sweep_ts.tzinfo is None:
            sweep_ts = sweep_ts.tz_localize(tz)
        df_after_sweep_1m = cur_nq.loc[sweep_ts : ts]
        if df_after_sweep_1m.empty or len(df_after_sweep_1m) < 10:
            continue

        active_fvgs = []
        for tf in entry_tfs:
            df_tf = _resample(df_after_sweep_1m, tf)
            z = detect_fvgs(df_tf, timeframe_min=tf, min_gap_points=min_gap, max_age_min=max_age)
            active_fvgs.extend(z)

        if not active_fvgs:
            continue

        # Evaluate iFVG on 1m reaction candles.
        desired_side = "SHORT" if sweep_event.side == "BUY" else "LONG"

        # prioritize newest zones first
        active_fvgs_sorted = sorted(active_fvgs, key=lambda z: (z.created_at, z.timeframe_min), reverse=True)
        for fvg in active_fvgs_sorted:
            side = detect_ifvg(df=df_after_sweep_1m, fvg=fvg, reaction_candles=reaction_candles)
            if side is None:
                continue
            if side != desired_side:
                continue

            entry_px = float(cur_nq.iloc[-1]["close"])
            entry_plan = build_entry_plan(
                side=side,
                entry=entry_px,
                sweep_price=float(sweep_event.sweep_price),
                stop_buffer=stop_buffer,
                rr=rr,
                timeframe_min=int(fvg.timeframe_min),
                orb_high=res.orb_high,
                orb_low=res.orb_low,
            )
            entry_ts = ts
            res.entry = {
                "ts": str(ts),
                "side": entry_plan.side,
                "entry": entry_plan.entry,
                "stop": entry_plan.stop,
                "target_2r": entry_plan.target,
                "timeframe_min": entry_plan.timeframe_min,
                "reason": entry_plan.reason,
                "fvg": asdict(fvg),
            }
            break

        if entry_plan is not None:
            break

    if sweep_event is None:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_SWEEP"
        return res

    if not active_fvgs:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_FVG_AFTER_SWEEP"
        return res

    if entry_plan is None or entry_ts is None:
        res.outcome = "SKIP"
        res.skipped_reason = "NO_IFVG_CONFIRMATION"
        return res

    # Session extremes up to entry (developing)
    session_hi_at_entry = float(nq_1m.loc[:entry_ts]["high"].max())
    session_lo_at_entry = float(nq_1m.loc[:entry_ts]["low"].min())

    exit_info = _simulate_trade(
        df_1m=nq_1m,
        entry_plan=entry_plan,
        entry_ts=entry_ts,
        prev_high=prev_high,
        prev_low=prev_low,
        session_hi_at_entry=session_hi_at_entry,
        session_lo_at_entry=session_lo_at_entry,
        eod_ts=end_ts,
    )
    res.exit = exit_info

    exit_px = float(exit_info["exit_price"])
    pnl_points = (exit_px - entry_plan.entry) if entry_plan.side == "LONG" else (entry_plan.entry - exit_px)
    res.pnl_points = float(pnl_points)
    res.pnl_dollars = float(pnl_points * NQ_DOLLARS_PER_POINT)

    if abs(res.pnl_points) < 1e-9:
        res.outcome = "BREAKEVEN"
    elif res.pnl_points > 0:
        res.outcome = "WIN"
    else:
        res.outcome = "LOSS"

    return res


def summarize(results: List[DayResult]) -> Dict[str, Any]:
    analyzed = len(results)
    setups = [r for r in results if r.outcome != "SKIP"]

    wins = [r for r in setups if r.outcome == "WIN"]
    losses = [r for r in setups if r.outcome == "LOSS"]
    bes = [r for r in setups if r.outcome == "BREAKEVEN"]

    total_pnl = sum(r.pnl_points for r in setups)
    total_pnl_dollars = sum(r.pnl_dollars for r in setups)

    avg_win = (sum(r.pnl_points for r in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(r.pnl_points for r in losses) / len(losses)) if losses else 0.0

    gross_profit = sum(r.pnl_points for r in wins)
    gross_loss = abs(sum(r.pnl_points for r in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    largest_win = max((r.pnl_points for r in setups), default=0.0)
    largest_loss = min((r.pnl_points for r in setups), default=0.0)

    win_rate = (len(wins) / len(setups) * 100.0) if setups else 0.0

    return {
        "days_analyzed": analyzed,
        "days_with_setups": len(setups),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(bes),
        "win_rate_pct": win_rate,
        "avg_win_points": avg_win,
        "avg_loss_points": avg_loss,
        "avg_win_dollars": avg_win * NQ_DOLLARS_PER_POINT,
        "avg_loss_dollars": avg_loss * NQ_DOLLARS_PER_POINT,
        "profit_factor": profit_factor,
        "total_pnl_points": total_pnl,
        "total_pnl_dollars": total_pnl_dollars,
        "largest_win_points": largest_win,
        "largest_loss_points": largest_loss,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest ORB + sweep + iFVG strategy")
    ap.add_argument("--days", type=int, default=30, help="Number of most-recent trading days to test")
    ap.add_argument("--settings", type=str, default="config/settings.yaml", help="Path to settings.yaml")
    ap.add_argument("--secrets", type=str, default="config/secrets.yaml", help="Path to secrets.yaml")
    ap.add_argument("--no-interactive-auth", action="store_true", help="Fail if token is missing/expired")
    ap.add_argument("--out", type=str, default="logs/backtest_results.json", help="Output JSON path")
    args = ap.parse_args()

    settings = _load_settings(args.settings)
    symbols = settings.get("symbols", {})
    nq_symbol = str(symbols.get("primary", "/NQ"))
    es_symbol = str(symbols.get("smt_compare", "/ES"))

    client = SchwabClient(secrets_path=args.secrets)
    ok = client.authenticate(interactive=(not args.no_interactive_auth))
    if not ok:
        logger.error("Schwab auth failed (non-interactive)")
        return 2

    # Use yesterday as end day to avoid partial current session.
    end_day = (datetime.now().date() - timedelta(days=1))
    days = _trading_days(end_day, int(args.days))

    results: List[DayResult] = []
    for d in days:
        try:
            r = backtest_day(client=client, day=d, settings=settings, nq_symbol=nq_symbol, es_symbol=es_symbol)
        except Exception as e:
            logger.exception(f"Day {d} failed: {e}")
            r = DayResult(day=str(d), analyzed=False, skipped_reason=f"ERROR: {e}")
        results.append(r)

    summary = summarize(results)

    print("\n=== ORB + Sweep + iFVG Backtest Summary ===")
    print(f"Days analyzed:      {summary['days_analyzed']}")
    print(f"Days with setups:   {summary['days_with_setups']}")
    print(f"Wins / Loss / BE:   {summary['wins']} / {summary['losses']} / {summary['breakeven']}")
    print(f"Win rate:           {summary['win_rate_pct']:.1f}%")
    print(f"Avg win (pts/$):    {summary['avg_win_points']:.2f} / {summary['avg_win_dollars']:.2f}")
    print(f"Avg loss (pts/$):   {summary['avg_loss_points']:.2f} / {summary['avg_loss_dollars']:.2f}")
    print(f"Profit factor:      {summary['profit_factor']:.2f}")
    print(f"Total PnL (pts/$):  {summary['total_pnl_points']:.2f} / {summary['total_pnl_dollars']:.2f}")
    print(f"Largest win/loss:   {summary['largest_win_points']:.2f} / {summary['largest_loss_points']:.2f}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "settings_path": args.settings,
        "symbols": {"nq": nq_symbol, "es": es_symbol},
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=DateTimeEncoder))
    print(f"Saved detailed results -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
