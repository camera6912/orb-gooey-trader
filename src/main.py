"""ORB + Gooey Model signal bot (paper trading).

State machine:
- WAITING (pre-9:30)
- ORB_BUILDING (9:30–9:45)
- WATCHING_SWEEP (post-ORB)
- WATCHING_FVG (after sweep)
- WAITING_IFVG (after FVG)
- IN_POSITION (paper position)
- DONE (after exit or 16:00)

This implementation focuses on *core logic + logging*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

from src.data.schwab import SchwabClient
from src.notifications.alerts import (
    ExitSummary,
    format_fvg,
    format_ifvg_entry,
    format_orb_set,
    format_smt,
    format_sweep,
)
from src.notifications.campfire import notifier_from_config
from src.strategy.fvg import FVG, detect_fvgs, detect_ifvg
from src.strategy.liquidity import SweepEvent, detect_sweep
from src.strategy.orb import ORBTracker, OpeningRange
from src.strategy.signals import EntryPlan, build_entry_plan
from src.strategy.smt import SMTSignal, detect_smt


class State(str, Enum):
    WAITING = "WAITING"
    ORB_BUILDING = "ORB_BUILDING"
    WATCHING_SWEEP = "WATCHING_SWEEP"
    WATCHING_FVG = "WATCHING_FVG"
    WAITING_IFVG = "WAITING_IFVG"
    IN_POSITION = "IN_POSITION"
    DONE = "DONE"


@dataclass
class PaperPosition:
    side: str
    entry: float
    stop: float
    target: float
    opened_at: datetime
    timeframe_min: int


def _parse_hhmm(s: str) -> dtime:
    hh, mm = [int(x) for x in str(s).split(":")]
    return dtime(hour=hh, minute=mm)


def _today_at(t: dtime, tz: str = "America/New_York") -> datetime:
    now = datetime.now().astimezone()
    # Use local timezone for simplicity; Schwab candles are converted to NY in client.
    # If host tz differs, you can make this explicit with zoneinfo.
    return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _setup_logging(settings: dict) -> None:
    log_cfg = settings.get("logging") or {}
    level = log_cfg.get("level", "INFO")
    file_path = log_cfg.get("file", "logs/orb-gooey-trader.log")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    logger.add(file_path, level=level, rotation="10 MB", retention=10)


def _now_et() -> datetime:
    # Schwab data uses America/New_York; treat system as ET for now.
    return datetime.now().astimezone()


def main() -> None:
    settings = _load_yaml("config/settings.yaml")
    try:
        secrets = _load_yaml("config/secrets.yaml")
    except FileNotFoundError:
        secrets = {}

    _setup_logging(settings)

    sym_nq = settings.get("symbols", {}).get("primary", "/NQ")
    sym_es = settings.get("symbols", {}).get("smt_compare", "/ES")

    session_cfg = settings.get("session") or {}
    orb_start_t = _parse_hhmm(session_cfg.get("orb_start", "09:30"))
    orb_end_t = _parse_hhmm(session_cfg.get("orb_end", "09:45"))
    eod_t = _parse_hhmm(session_cfg.get("eod_exit", "16:00"))

    strat = settings.get("strategy") or {}
    fvg_cfg = strat.get("fvg") or {}
    liq_cfg = strat.get("liquidity") or {}
    risk_cfg = strat.get("risk") or {}

    entry_tfs = [int(x) for x in (fvg_cfg.get("entry_timeframes_min") or [1, 3, 5])]

    notifier = notifier_from_config(settings, secrets)

    sc = SchwabClient("config/secrets.yaml")
    sc.authenticate(interactive=True)

    orb = ORBTracker()

    state: State = State.WAITING
    current_orb: Optional[OpeningRange] = None
    sweep: Optional[SweepEvent] = None
    smt_sig: Optional[SMTSignal] = None
    chosen_fvg: Optional[FVG] = None
    position: Optional[PaperPosition] = None

    logger.info(f"Starting orb-gooey-trader for {sym_nq} (SMT vs {sym_es})")

    while True:
        now = _now_et()

        orb_start = _today_at(orb_start_t)
        orb_end = _today_at(orb_end_t)
        eod = _today_at(eod_t)

        if now >= eod and state != State.DONE:
            logger.info("EOD reached; moving to DONE")
            state = State.DONE

        if state == State.DONE:
            logger.info("DONE. Exiting.")
            return

        # Fetch 1m candles for NQ/ES (shared across states)
        try:
            nq_1m = sc.get_intraday(sym_nq, 1, bars=800)
            es_1m = sc.get_intraday(sym_es, 1, bars=800)
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            time.sleep(10)
            continue

        last_price = float(nq_1m.iloc[-1]["close"]) if not nq_1m.empty else 0.0

        # State transitions
        if state == State.WAITING:
            if now >= orb_start and now < orb_end:
                logger.info("State: WAITING -> ORB_BUILDING")
                orb.start(orb_start, orb_end)
                state = State.ORB_BUILDING
            else:
                time.sleep(5)
                continue

        if state == State.ORB_BUILDING:
            # Update ORB from candles in window
            orb_window = nq_1m[(nq_1m.index >= orb_start) & (nq_1m.index <= now)]
            if not orb_window.empty:
                orb.update(price_high=float(orb_window["high"].max()), price_low=float(orb_window["low"].min()))

            if now >= orb_end:
                try:
                    current_orb = orb.finalize()
                except Exception as e:
                    logger.error(f"Failed to finalize ORB: {e}")
                    time.sleep(5)
                    continue

                msg = format_orb_set(symbol=sym_nq, high=current_orb.high, low=current_orb.low, start=current_orb.start, end=current_orb.end)
                logger.info(msg)
                notifier.send_message(msg)

                logger.info("State: ORB_BUILDING -> WATCHING_SWEEP")
                state = State.WATCHING_SWEEP

            time.sleep(5)
            continue

        if state == State.WATCHING_SWEEP:
            assert current_orb is not None

            sweep = detect_sweep(
                df_1m=nq_1m,
                orb_high=current_orb.high,
                orb_low=current_orb.low,
                buffer_points=float(liq_cfg.get("sweep_buffer_points", 0.5)),
                require_close_back_inside=bool(liq_cfg.get("sweep_confirm_close_back_inside", True)),
            )

            if sweep:
                # Determine expected reversal side
                # BUY sweep (above highs) => reversal SHORT; SELL sweep => reversal LONG
                rev_side = "SHORT" if sweep.side == "BUY" else "LONG"
                msg = format_sweep(symbol=sym_nq, side=rev_side, sweep_price=sweep.sweep_price, orb_high=current_orb.high, orb_low=current_orb.low)
                logger.info(msg)
                notifier.send_message(msg)

                # SMT confirmation (optional gate)
                smt_sig = detect_smt(nq=nq_1m, es=es_1m, lookback=int(liq_cfg.get("swing_lookback", 20)))
                if smt_sig:
                    smt_msg = format_smt(smt=f"{smt_sig.smt} leader={smt_sig.leader}")
                    logger.info(smt_msg)
                    notifier.send_message(smt_msg)

                logger.info("State: WATCHING_SWEEP -> WATCHING_FVG")
                state = State.WATCHING_FVG

            time.sleep(5)
            continue

        if state == State.WATCHING_FVG:
            assert current_orb is not None and sweep is not None

            # Scan for most recent FVG across configured timeframes.
            fvgs: list[FVG] = []
            for tf in entry_tfs:
                df_tf = sc.get_intraday(sym_nq, tf, bars=400)
                fvgs.extend(
                    detect_fvgs(
                        df_tf,
                        timeframe_min=tf,
                        min_gap_points=float(fvg_cfg.get("min_gap_points", 0.25)),
                        max_age_min=int(fvg_cfg.get("max_fvg_age_min", 120)),
                    )
                )

            if fvgs:
                # Choose the newest FVG that aligns with sweep direction (we want opposite)
                fvgs = sorted(fvgs, key=lambda z: z.created_at, reverse=True)

                expected_side = "LONG" if sweep.side == "SELL" else "SHORT"

                for z in fvgs:
                    if expected_side == "LONG" and z.fvg_type == "BEAR":
                        chosen_fvg = z
                        break
                    if expected_side == "SHORT" and z.fvg_type == "BULL":
                        chosen_fvg = z
                        break

                # If none align, still take newest as a starting point
                if chosen_fvg is None:
                    chosen_fvg = fvgs[0]

                msg = format_fvg(
                    timeframe_min=chosen_fvg.timeframe_min,
                    fvg_type=chosen_fvg.fvg_type,
                    low=chosen_fvg.low,
                    high=chosen_fvg.high,
                    created_at=chosen_fvg.created_at,
                )
                logger.info(msg)
                notifier.send_message(msg)

                logger.info("State: WATCHING_FVG -> WAITING_IFVG")
                state = State.WAITING_IFVG

            time.sleep(5)
            continue

        if state == State.WAITING_IFVG:
            assert current_orb is not None and sweep is not None and chosen_fvg is not None

            df_tf = sc.get_intraday(sym_nq, chosen_fvg.timeframe_min, bars=400)
            entry_side = detect_ifvg(
                df=df_tf,
                fvg=chosen_fvg,
                reaction_candles=int(fvg_cfg.get("ifvg_reaction_candles", 2)),
            )

            if entry_side:
                entry = float(df_tf.iloc[-1]["close"])
                plan: EntryPlan = build_entry_plan(
                    side=entry_side,
                    entry=entry,
                    sweep_price=sweep.sweep_price,
                    stop_buffer=float(risk_cfg.get("stop_buffer_points", 0.5)),
                    rr=float(risk_cfg.get("default_rr", 2.0)),
                    timeframe_min=chosen_fvg.timeframe_min,
                    orb_high=current_orb.high,
                    orb_low=current_orb.low,
                )

                msg = format_ifvg_entry(
                    symbol=sym_nq,
                    side=plan.side,
                    entry=plan.entry,
                    stop=plan.stop,
                    target=plan.target,
                    timeframe_min=plan.timeframe_min,
                    reason=plan.reason,
                )
                logger.info(msg)
                notifier.send_message(msg)

                position = PaperPosition(
                    side=plan.side,
                    entry=plan.entry,
                    stop=plan.stop,
                    target=plan.target,
                    opened_at=now,
                    timeframe_min=plan.timeframe_min,
                )

                logger.info("State: WAITING_IFVG -> IN_POSITION")
                state = State.IN_POSITION

            time.sleep(5)
            continue

        if state == State.IN_POSITION:
            assert position is not None

            # Simple position management on last_price
            hit_target = last_price >= position.target if position.side == "LONG" else last_price <= position.target
            hit_stop = last_price <= position.stop if position.side == "LONG" else last_price >= position.stop

            if hit_target or hit_stop or now >= eod:
                reason = "target" if hit_target else ("stop" if hit_stop else "eod")
                pnl_pts = (last_price - position.entry) if position.side == "LONG" else (position.entry - last_price)
                dur_s = (now - position.opened_at).total_seconds()
                summary = ExitSummary(exit_reason=reason, pnl_points=float(pnl_pts), duration_s=float(dur_s))
                from src.notifications.alerts import format_exit

                msg = format_exit(symbol=sym_nq, side=position.side, entry=position.entry, exit_price=last_price, summary=summary)
                logger.info(msg)
                notifier.send_message(msg)

                logger.info("State: IN_POSITION -> DONE")
                state = State.DONE
                continue

            time.sleep(5)
            continue


if __name__ == "__main__":
    main()
