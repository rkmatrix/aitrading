# ai/market/data_provider.py
from __future__ import annotations
import os, json, time, random, logging, urllib.request
from typing import Dict, List

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class MarketDataProvider:
    """
    Minimal price provider:
      - tries Alpaca last trade API if keys exist
      - otherwise returns a drifting fake price
    """
    def __init__(self, symbols: List[str], cfg: dict | None = None):
        self.symbols = symbols
        self.cfg = cfg or {}
        self._fake_prices: Dict[str, float] = {s: 100.0 + random.uniform(-5, 5) for s in symbols}

        # Alpaca env
        self._alpaca_key = os.getenv(self.cfg.get("alpaca", {}).get("key_env", "APCA_API_KEY_ID"), "")
        self._alpaca_secret = os.getenv(self.cfg.get("alpaca", {}).get("secret_env", "APCA_API_SECRET_KEY"), "")
        self._base_url = os.getenv("APCA_API_BASE_URL", self.cfg.get("alpaca", {}).get("base_url", "https://paper-api.alpaca.markets"))

        # If user provided price_source=fake force fake
        self._force_fake = (self.cfg.get("price_source") == "fake")

    def _get_alpaca_last_quote(self, symbol: string) -> float:
        # Use Alpaca v2 last trade endpoint
        # Note: Some accounts need "data" API base. Keep fallback robust.
        data_url = os.getenv("APCA_DATA_API_BASE_URL", "https://data.alpaca.markets")
        url = f"{data_url}/v2/stocks/{symbol}/trades/latest"
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        px = float(payload.get("trade", {}).get("p") or 0.0)
        return px

    def prices(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._force_fake or not (self._alpaca_key and self._alpaca_secret):
            # fake drift
            for s in self.symbols:
                p = self._fake_prices[s]
                p *= (1.0 + random.uniform(-0.002, 0.002))
                p = max(p, 1.0)
                self._fake_prices[s] = p
                out[s] = p
            return out

        # try Alpaca
        for s in self.symbols:
            try:
                p = self._get_alpaca_last_quote(s)
                if p <= 0:
                    raise ValueError("Zero price")
                out[s] = p
            except Exception as e:
                logger.warning("Price fetch fallback for %s (%s)", s, e)
                # drift fake for just this symbol
                p = self._fake_prices[s] * (1.0 + random.uniform(-0.002, 0.002))
                self._fake_prices[s] = max(p, 1.0)
                out[s] = self._fake_prices[s]
        return out
