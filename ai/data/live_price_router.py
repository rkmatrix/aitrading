"""
ai/data/live_price_router.py

Phase 79 – LivePriceRouter

Fetches latest OHLCV bar for a symbol using provider priority:
    1) Alpaca  (if APCA_API_KEY_ID/APCA_API_SECRET_KEY set)
    2) Polygon (if POLYGON_API_KEY set)
    3) TwelveData (if TWELVEDATA_API_KEY set)

Returns a unified dict:
    {
        "symbol": str,
        "ts": datetime,
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
    }
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class LivePriceRouter:
    """
    Multi-provider live bar router with automatic failover.

    Usage:
        router = LivePriceRouter()
        bar = router.get_latest_bar("AAPL")
        if bar:
            price = bar["close"]
    """

    def __init__(self, bar_interval: str = "1min") -> None:
        self.bar_interval = bar_interval

        self.apca_key = os.getenv("APCA_API_KEY_ID")
        self.apca_secret = os.getenv("APCA_API_SECRET_KEY")

        self.polygon_key = os.getenv("POLYGON_API_KEY")
        self.twelvedata_key = os.getenv("TWELVEDATA_API_KEY")

        # Provider priority: Alpaca → Polygon → TwelveData
        self.providers = []
        if self.apca_key and self.apca_secret:
            self.providers.append("alpaca")
        if self.polygon_key:
            self.providers.append("polygon")
        if self.twelvedata_key:
            self.providers.append("twelvedata")

        if not self.providers:
            logger.warning(
                "LivePriceRouter: No live data providers configured. "
                "Set APCA_API_KEY_ID / POLYGON_API_KEY / TWELVEDATA_API_KEY."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest OHLCV bar for the given symbol from the first
        working provider: Alpaca → Polygon → TwelveData.

        Returns None if all providers fail.
        """
        for provider in self.providers:
            try:
                if provider == "alpaca":
                    bar = self._from_alpaca(symbol)
                elif provider == "polygon":
                    bar = self._from_polygon(symbol)
                elif provider == "twelvedata":
                    bar = self._from_twelvedata(symbol)
                else:
                    continue

                if bar is not None:
                    logger.info("LivePriceRouter: Using %s for %s", provider, symbol)
                    return bar

            except Exception as e:
                logger.warning(
                    "LivePriceRouter: Provider %s failed for %s → %s",
                    provider,
                    symbol,
                    e,
                )

        logger.error("LivePriceRouter: All providers failed for %s", symbol)
        return None

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    def _from_alpaca(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Use Alpaca Market Data v2 latest bars endpoint.

        Docs: https://docs.alpaca.markets/reference/stocklatestbars
        """
        base_url = "https://data.alpaca.markets/v2/stocks/bars/latest"
        headers = {
            "APCA-API-KEY-ID": self.apca_key,
            "APCA-API-SECRET-KEY": self.apca_secret,
        }
        params = {"symbols": symbol.upper()}
        resp = requests.get(base_url, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Alpaca HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        bars = data.get("bars", {})
        sb = bars.get(symbol.upper())
        if not sb:
            return None

        # Alpaca returns timestamps as RFC3339
        ts_raw = sb.get("t")
        ts = (
            datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if ts_raw
            else datetime.utcnow()
        )

        return {
            "symbol": symbol.upper(),
            "ts": ts,
            "open": float(sb["o"]),
            "high": float(sb["h"]),
            "low": float(sb["l"]),
            "close": float(sb["c"]),
            "volume": float(sb["v"]),
        }

    def _from_polygon(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Use Polygon previous-day aggregate as a fallback.
        Not truly real-time, but gives a valid OHLCV bar if Alpaca fails.

        Docs pattern:
            GET https://api.polygon.io/v2/aggs/ticker/{symbol}/prev
        """
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/prev"
        params = {"adjusted": "true", "apiKey": self.polygon_key}
        resp = requests.get(base_url, params=params, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Polygon HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        results = data.get("results") or []
        if not results:
            return None

        r = results[0]
        ts = datetime.utcfromtimestamp(r["t"] / 1000.0)

        return {
            "symbol": symbol.upper(),
            "ts": ts,
            "open": float(r["o"]),
            "high": float(r["h"]),
            "low": float(r["l"]),
            "close": float(r["c"]),
            "volume": float(r["v"]),
        }

    def _from_twelvedata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Use TwelveData intraday time_series with outputsize=1.

        Docs: https://twelvedata.com/docs
        """
        base_url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol.upper(),
            "interval": self.bar_interval,
            "outputsize": 1,
            "apikey": self.twelvedata_key,
        }
        resp = requests.get(base_url, params=params, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"TwelveData HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        if "values" not in data or not data["values"]:
            return None

        v = data["values"][0]
        ts = datetime.fromisoformat(v["datetime"])

        return {
            "symbol": symbol.upper(),
            "ts": ts,
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
            "volume": float(v.get("volume", 0.0)),
        }
