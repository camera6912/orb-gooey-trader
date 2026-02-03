"""Schwab market data wrapper for /NQ + /ES.

This is based on the Schwab client used in `~/projects/orb-trader`, adapted for:
- futures quotes for /NQ and /ES
- intraday candle fetch + resampling to multiple timeframes

No order endpoints are used (paper/signal-only).

Requires: schwab-py (https://github.com/alexgolec/schwab-py)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

try:
    from schwab import auth
    from schwab.client import Client

    SCHWAB_AVAILABLE = True
except Exception:
    SCHWAB_AVAILABLE = False


@dataclass(frozen=True)
class FuturesQuote:
    requested_symbol: str  # e.g. /NQ
    resolved_symbol: str   # e.g. /NQH26
    last: float
    bid: float
    ask: float
    high: float
    low: float
    open: float
    close: float
    volume: int
    ts: Optional[datetime] = None


class SchwabClient:
    def __init__(self, secrets_path: str = "config/secrets.yaml"):
        if not SCHWAB_AVAILABLE:
            raise ImportError("schwab-py required. Install: pip install schwab-py")

        self.secrets = self._load_secrets(secrets_path)
        self.client: Optional[Client] = None
        self.token_path = Path(self.secrets.get("token_path", "./config/schwab_token.json"))

    def _load_secrets(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Secrets not found: {path} (copy secrets.yaml.example -> secrets.yaml)")
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        return (raw.get("schwab") or {})

    def authenticate(self, interactive: bool = True) -> bool:
        app_key = self.secrets["app_key"]
        app_secret = self.secrets["app_secret"]
        callback_url = self.secrets["callback_url"]

        try:
            if self.token_path.exists():
                self.client = auth.client_from_token_file(
                    token_path=str(self.token_path),
                    api_key=app_key,
                    app_secret=app_secret,
                )
                logger.info("Schwab authenticated with cached token")
                return True
        except Exception as e:
            logger.warning(f"Cached token invalid: {e}")

        if not interactive:
            return False

        logger.info("Starting Schwab OAuth manual flow...")
        self.client = auth.client_from_manual_flow(
            api_key=app_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=str(self.token_path),
        )
        logger.info("Schwab authentication successful")
        return True

    def get_quotes_raw(self, symbols: List[str]) -> dict:
        if not self.client:
            raise RuntimeError("Not authenticated")

        resp = self.client.get_quotes(symbols)
        if resp.status_code != 200:
            raise RuntimeError(f"Quotes failed: {resp.status_code} {resp.text}")
        return resp.json() or {}

    def get_futures_quotes(self, symbols: List[str]) -> Dict[str, FuturesQuote]:
        """Returns dict keyed by requested symbol (/NQ, /ES)."""
        raw = self.get_quotes_raw(symbols)
        out: Dict[str, FuturesQuote] = {}

        # Schwab resolves /ES to front month like /ESH26; same for /NQ.
        # We map by checking which resolved symbol starts with the requested root.
        for req in symbols:
            root = req.strip().upper()
            match_sym = None
            match_data = None
            for sym, data in raw.items():
                if str(sym).upper().startswith(root):
                    match_sym, match_data = sym, data
                    break
            if not match_data:
                continue

            q = match_data.get("quote", {})
            out[root] = FuturesQuote(
                requested_symbol=root,
                resolved_symbol=str(match_sym),
                last=float(q.get("lastPrice") or 0.0),
                bid=float(q.get("bidPrice") or 0.0),
                ask=float(q.get("askPrice") or 0.0),
                high=float(q.get("highPrice") or 0.0),
                low=float(q.get("lowPrice") or 0.0),
                open=float(q.get("openPrice") or 0.0),
                close=float(q.get("closePrice") or 0.0),
                volume=int(q.get("totalVolume") or 0),
            )

        return out

    def get_intraday_1m(self, symbol: str, bars: int = 600) -> pd.DataFrame:
        """Fetch 1m candles for the current day.

        Schwab's intraday history API is easiest to consume at 1m and then resample.
        `bars` limits how many rows we keep after fetch.
        """
        if not self.client:
            raise RuntimeError("Not authenticated")

        resp = self.client.get_price_history(
            symbol=symbol,
            period_type=self.client.PriceHistory.PeriodType.DAY,
            period=self.client.PriceHistory.Period.ONE_DAY,
            frequency_type=self.client.PriceHistory.FrequencyType.MINUTE,
            frequency=self.client.PriceHistory.Frequency.EVERY_MINUTE,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"History failed: {resp.status_code} {resp.text}")

        data = resp.json() or {}
        candles = data.get("candles") or []
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        df = df.set_index("datetime").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        if bars and len(df) > bars:
            df = df.iloc[-bars:]
        return df

    def get_intraday(self, symbol: str, timeframe_min: int, bars: int = 600) -> pd.DataFrame:
        """Get intraday candles resampled to timeframe_min (1,3,5,15,240...)."""
        base = self.get_intraday_1m(symbol, bars=max(bars * max(1, timeframe_min), 1000))
        if base.empty:
            return base
        if timeframe_min == 1:
            return base.tail(bars)

        rule = f"{int(timeframe_min)}min"
        o = base["open"].resample(rule).first()
        h = base["high"].resample(rule).max()
        l = base["low"].resample(rule).min()
        c = base["close"].resample(rule).last()
        v = base["volume"].resample(rule).sum()
        df = pd.concat([o, h, l, c, v], axis=1)
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna().tail(bars)
        return df
