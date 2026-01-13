# ai/data/price_resolver.py
# -------------------------------------------------------------------
# Unified Multi-Vendor Price Resolver (Phase 9.2 + Alpha Vantage Free)
# -------------------------------------------------------------------
# • Priority: Cache → Alpha Vantage (free TIME_SERIES_DAILY) → Yahoo
# • Optional: Polygon, Finnhub, Alpaca
# • Performs SLA & data-quality checks (NaN ratio, lag, disagreement)
# • Writes fetched data back into data/cache/
# -------------------------------------------------------------------

from __future__ import annotations
import os
import time
import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Optional

import yfinance as yf
import requests

from app_config import (
    CFG,
    DATA_DISAGREEMENT_BPS,
    DATA_MAX_LAG_MIN,
    DATA_MAX_NAN_RATIO,
)
from utils.logger import log
from utils.config_loader import get_alpha_key

try:
    
except Exception:
    tradeapi = None


# -------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _bps_diff(a: float, b: float) -> float:
    if a == 0 or np.isnan(a) or np.isnan(b):
        return 0.0
    return abs((a - b) / a) * 1e4


# -------------------------------------------------------------------
# PriceResolver
# -------------------------------------------------------------------

class PriceResolver:
    """Multi-vendor resolver with caching, Alpha Vantage fallback, and SLA checks."""

    def __init__(self):
        self.cache_dir = CFG.data.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.alpha_key = CFG.news_macro.alphavantage_key or get_alpha_key()
        self.finnhub_key = CFG.news_macro.finnhub_key
        self.polygon_key = CFG.news_macro.polygon_key
        self.alpaca_key = CFG.alpaca.key
        self.alpaca_secret = CFG.alpaca.secret
        self.cache_expiry_days = 1

        log(f"📦 PriceResolver initialized → cache={self.cache_dir}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_close_matrix(
        self, symbols: List[str], period: str = "365d", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Return a Close-price DataFrame indexed by datetime.
        Priority:
          1. Cache
          2. Alpha Vantage (free)
          3. Yahoo Finance
          4. Optional (Polygon/Finnhub/Alpaca)
        """
        log(f"🔍 Resolving {len(symbols)} symbols (period={period}, interval={interval})")

        frames: Dict[str, pd.Series] = {}
        failed: List[str] = []

        for sym in symbols:
            s = None

            # --- Cache
            s = self._from_cache(sym, period, interval)
            if s is not None and not s.empty:
                frames[sym] = s
                continue

            # --- Alpha Vantage
            s = self._from_alpha(sym)
            if s is not None and not s.empty:
                frames[sym] = s
                continue

            # --- Yahoo Finance
            s = self._from_yahoo(sym, period, interval)
            if s is not None and not s.empty:
                frames[sym] = s
                continue

            # --- Optional vendors
            s = self._from_optional(sym, period, interval)
            if s is not None and not s.empty:
                frames[sym] = s
                continue

            # --- All failed
            failed.append(sym)
            log(f"❌ No data found for {sym}")

        if failed:
            log(f"⚠️  {len(failed)} Failed downloads: {failed}")

        if not frames:
            log("❌ All providers failed; returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(frames).dropna(how="all")
        df = _normalize_df(df)
        self._check_data_quality(df)
        return df

    # ------------------------------------------------------------------
    # Cache layer
    # ------------------------------------------------------------------
    def _cache_path(self, sym: str, period: str, interval: str) -> str:
        return os.path.join(self.cache_dir, f"{sym}_{period}_{interval}.csv")

    def _from_cache(self, sym: str, period: str, interval: str) -> Optional[pd.Series]:
        path = self._cache_path(sym, period, interval)
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            age_days = (time.time() - mtime) / 86400
            if age_days > self.cache_expiry_days:
                log(f"💾 Cache expired for {sym} ({age_days:.1f} d)")
                return None
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "Close" not in df.columns:
                return None
            if len(df) == 0:
                return None
            log(f"💾 Cache hit for {sym} ({len(df)} rows)")
            return df["Close"]
        except Exception as e:
            log(f"⚠️  Cache read failed for {sym}: {e}")
            return None

    def _save_cache(self, sym: str, period: str, interval: str, s: pd.Series):
        try:
            path = self._cache_path(sym, period, interval)
            s.to_frame("Close").to_csv(path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Alpha Vantage (free TIME_SERIES_DAILY)
    # ------------------------------------------------------------------
    def _from_alpha(self, sym: str) -> Optional[pd.Series]:
        """Use Alpha Vantage free daily endpoint (works for free API keys)."""
        if not self.alpha_key:
            return None
        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=TIME_SERIES_DAILY&symbol={sym}&outputsize=full&apikey={self.alpha_key}"
            )
            r = requests.get(url, timeout=15)
            js = r.json()
            if "Time Series (Daily)" not in js:
                note = js.get("Note") or js.get("Information") or "unknown"
                log(f"⚠️  Alpha Vantage no data for {sym}: {note}")
                return None
            df = (
                pd.DataFrame(js["Time Series (Daily)"])
                .T.rename(columns={"4. close": "Close"})
                .astype(float)
            )
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            s = df["Close"]
            self._save_cache(sym, "730d", "1d", s)
            log(f"📡 Alpha Vantage (free) → {sym} ({len(df)} rows)")
            return s
        except Exception as e:
            log(f"⚠️  Alpha Vantage (free) failed for {sym}: {e}")
            return None

    # ------------------------------------------------------------------
    # Yahoo Finance
    # ------------------------------------------------------------------
    def _from_yahoo(self, sym: str, period: str, interval: str) -> Optional[pd.Series]:
        try:
            df = yf.download(
                sym, period=period, interval=interval, auto_adjust=True, progress=False
            )
            if not df.empty:
                df = _normalize_df(df)
                s = df["Close"]
                self._save_cache(sym, period, interval, s)
                log(f"🌐 Yahoo Finance → {sym} ({len(df)} rows)")
                return s
        except Exception as e:
            log(f"⚠️  Yahoo Finance failed for {sym}: {e}")
        return None

    # ------------------------------------------------------------------
    # Optional vendors (Polygon, Finnhub, Alpaca)
    # ------------------------------------------------------------------
    def _from_optional(self, sym: str, period: str, interval: str) -> Optional[pd.Series]:
        s = None
        if self.polygon_key:
            s = self._from_polygon(sym)
        if s is None and self.finnhub_key:
            s = self._from_finnhub(sym)
        if s is None and self.alpaca_key:
            s = self._from_alpaca(sym)
        return s

    def _from_polygon(self, sym: str) -> Optional[pd.Series]:
        try:
            end = dt.date.today()
            start = end - dt.timedelta(days=365)
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
                f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self.polygon_key}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                js = r.json()
                if "results" in js:
                    df = pd.DataFrame(js["results"])
                    df["t"] = pd.to_datetime(df["t"], unit="ms")
                    df = df.rename(columns={"c": "Close"}).set_index("t")
                    log(f"📊 Polygon → {sym} ({len(df)} rows)")
                    return df["Close"]
        except Exception as e:
            log(f"⚠️  Polygon failed for {sym}: {e}")
        return None

    def _from_finnhub(self, sym: str) -> Optional[pd.Series]:
        try:
            url = (
                f"https://finnhub.io/api/v1/stock/candle?symbol={sym}"
                f"&resolution=D&count=500&token={self.finnhub_key}"
            )
            r = requests.get(url, timeout=10)
            js = r.json()
            if js.get("s") == "ok":
                df = pd.DataFrame({"Close": js["c"]}, index=pd.to_datetime(js["t"], unit="s"))
                log(f"📈 Finnhub → {sym} ({len(df)} rows)")
                return df["Close"]
        except Exception as e:
            log(f"⚠️  Finnhub failed for {sym}: {e}")
        return None

    def _from_alpaca(self, sym: str) -> Optional[pd.Series]:
        if not tradeapi:
            return None
        try:
            api = tradeapi.REST(self.alpaca_key, self.alpaca_secret, CFG.alpaca.base_url)
            bars = api.get_bars(sym, "1Day", limit=500).df
            if not bars.empty:
                df = bars.rename(columns={"close": "Close"})
                log(f"📘 Alpaca → {sym} ({len(df)} rows)")
                return df["Close"]
        except Exception as e:
            log(f"⚠️  Alpaca failed for {sym}: {e}")
        return None

    # ------------------------------------------------------------------
    # Data-quality / SLA checks
    # ------------------------------------------------------------------
    def _check_data_quality(self, df: pd.DataFrame):
        try:
            nan_ratio = df.isna().sum().sum() / df.size
            if nan_ratio > DATA_MAX_NAN_RATIO:
                log(f"⚠️  High NaN ratio {nan_ratio:.3f}")

            last_ts = df.index[-1]
            age_min = (dt.datetime.utcnow() - last_ts.to_pydatetime()).total_seconds() / 60
            if age_min > DATA_MAX_LAG_MIN:
                log(f"⚠️  Data lag {age_min:.1f} min > SLA")

            last_row = df.iloc[-1].dropna()
            if len(last_row) >= 2:
                vals = last_row.values
                diffs = [
                    _bps_diff(vals[i], vals[j])
                    for i in range(len(vals))
                    for j in range(i + 1, len(vals))
                ]
                if np.max(diffs) > DATA_DISAGREEMENT_BPS:
                    log(f"⚠️  Cross-vendor disagreement {np.max(diffs):.1f} bps")
        except Exception as e:
            log(f"⚠️  Data-quality check failed: {e}")


# -------------------------------------------------------------------
# Stand-alone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    resolver = PriceResolver()
    syms = ["AAPL", "MSFT", "SPY"]
    df = resolver.get_close_matrix(syms, "730d", "1d")
    print(df.tail())
    print("Shape:", df.shape)
