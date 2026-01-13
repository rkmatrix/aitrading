"""
market_feed.py – Market data provider abstraction
Supports Alpaca (LIVE / PAPER) or synthetic simulator.
"""

from __future__ import annotations
import os
import random
import time
from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger("MarketFeed")

try:
    
except ImportError:
    tradeapi = None


class MarketFeed:
    def __init__(self, symbols: List[str], paper: bool = True):
        self.symbols = symbols
        self.paper = paper
        self.api = None
        self.connected = False
        self.last_prices: Dict[str, float] = {s: 100.0 for s in symbols}

        if tradeapi is not None:
            key = os.getenv("ALPACA_API_KEY")
            secret = os.getenv("ALPACA_SECRET_KEY")
            endpoint = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
            if key and secret:
                try:
                    self.api = tradeapi.REST(key, secret, base_url=endpoint)
                    self.connected = True
                    logger.info(f"📡 Alpaca feed connected (paper={paper})")
                except Exception as e:
                    logger.warning(f"⚠️  Alpaca connection failed: {e}")
        else:
            logger.warning("⚠️  alpaca_trade_api not installed – using random prices.")

    # ------------------------------------------------------------
    def reset(self):
        """Reset feed state."""
        for s in self.symbols:
            self.last_prices[s] = 100.0

    # ------------------------------------------------------------
    def last_price(self, symbol: str) -> float:
        """Return latest price for symbol (API or random)."""
        if self.connected and self.api:
            try:
                barset = self.api.get_latest_trade(symbol)
                p = float(barset.price)
                self.last_prices[symbol] = p
                return p
            except Exception as e:
                logger.warning(f"Fetch price failed for {symbol}: {e}")

        # fallback
        p = self.last_prices.get(symbol, 100.0) * (1 + np.random.randn() * 0.002)
        self.last_prices[symbol] = max(1.0, p)
        return self.last_prices[symbol]
