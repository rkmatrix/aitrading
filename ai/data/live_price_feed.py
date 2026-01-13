# ai/data/live_price_feed.py
"""
Simulated live price feed for DEMO/TEST modes.

This version is tuned to:
    • Generate more frequent BUY/SELL signals
    • Create clearer short/long MA crossovers
    • Produce noticeable volatility

Used by Phase 26+ realtime demo when real broker feeds are not wired.
"""

from __future__ import annotations

import asyncio
import random
from typing import Dict, List


class LivePriceFeed:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols

        # Start prices in a reasonable range
        self._price: Dict[str, float] = {s: random.uniform(100, 300) for s in symbols}
        self._last_price: Dict[str, float] = dict(self._price)

        # Moving averages start at price
        self._short_ma: Dict[str, float] = dict(self._price)
        self._long_ma: Dict[str, float] = dict(self._price)

        # Volatility estimate
        self._vol: Dict[str, float] = {s: 0.2 for s in symbols}

        self._tick_event = asyncio.Event()

    async def connect(self):
        """
        Starts a background task that updates prices every second.
        This simulates a live feed.
        """
        asyncio.create_task(self._price_loop())

    async def wait_for_new_tick(self):
        await self._tick_event.wait()
        self._tick_event.clear()

    async def _price_loop(self):
        """
        Simulates price dynamics with reasonably strong drift and noise so that:
            • short_ma moves quickly
            • long_ma moves slowly
            • volatility varies
        """
        while True:
            for sym in self.symbols:
                old = self._price[sym]

                # Stronger random drift to generate directional moves
                # e.g., up to ±2% per tick
                pct_drift = random.uniform(-0.02, 0.02)
                new = max(1.0, old * (1.0 + pct_drift))

                self._last_price[sym] = old
                self._price[sym] = new

                # Short MA: react faster to price
                self._short_ma[sym] = 0.4 * self._short_ma[sym] + 0.6 * new

                # Long MA: react slower
                self._long_ma[sym] = 0.95 * self._long_ma[sym] + 0.05 * new

                # Volatility: derived from MA divergence
                spread = abs(self._short_ma[sym] - self._long_ma[sym])
                base = max(self._long_ma[sym], 1e-6)
                vol_est = spread / base
                self._vol[sym] = max(0.05, min(0.8, vol_est))

            self._tick_event.set()
            await asyncio.sleep(1.0)

    # ---------------------------------------------------------------------
    # Exposed API
    # ---------------------------------------------------------------------

    def last_price(self, sym: str) -> float:
        return float(self._price.get(sym, 0.0))

    def price_drift(self, sym: str) -> float:
        old = self._last_price.get(sym, 0.0)
        new = self._price.get(sym, 0.0)
        if old <= 0:
            return 0.0
        return (new - old) / old

    def short_ma(self, sym: str) -> float:
        return float(self._short_ma.get(sym, 0.0))

    def long_ma(self, sym: str) -> float:
        return float(self._long_ma.get(sym, 0.0))

    def volatility(self, sym: str) -> float:
        return float(self._vol.get(sym, 0.2))

    # Router scoring (still dummy but more varied)
    def router_scores(self) -> Dict[str, float]:
        return {
            "alpaca": random.uniform(0.3, 1.0),
            "backup": random.uniform(0.0, 0.9),
        }
