"""
LiveMarketFeed
--------------
Unified interface for market price streaming.

Modes supported:
    - "ALPACA": real-time websocket stream from Alpaca Data API
    - "SIM": random-simulated feed (offline)
"""

import os, asyncio, random, logging, datetime as dt
import pandas as pd

try:
    from alpaca.data.live import StockDataStream
except Exception:
    StockDataStream = None

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
except Exception:
    pass

# ------------------------------------------------------------
class LiveMarketFeed:
    def __init__(self, symbols, mode="ALPACA"):
        self.symbols = symbols
        self.mode = mode.upper()
        self.stream = None
        self.listeners = []
        self._latest = {}

        # Load credentials
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_API_SECRET")

        logging.info(f"📡 LiveMarketFeed mode={self.mode} symbols={self.symbols}")

        if self.mode == "ALPACA" and StockDataStream:
            self._init_alpaca_stream()
        else:
            logging.warning("⚠️ Alpaca streaming unavailable; using SIM feed.")
            self.mode = "SIM"

    # ------------------------------------------------------------
    def _init_alpaca_stream(self):
        """Initialize Alpaca websocket stream."""
        if not (self.api_key and self.secret_key):
            raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment")

        self.stream = StockDataStream(self.api_key, self.secret_key)

        async def on_bar(bar):
            sym = bar.symbol
            self._latest[sym] = {
                "price": bar.close,
                "volume": bar.volume,
                "ts": bar.timestamp,
            }
            for cb in self.listeners:
                await cb(sym, self._latest[sym])

        for s in self.symbols:
            self.stream.subscribe_bars(on_bar, s)
        logging.info(f"✅ Alpaca live stream subscribed to {len(self.symbols)} symbols")

    # ------------------------------------------------------------
    async def stream_prices(self):
        """Async generator yielding dict of symbol->price."""
        if self.mode == "ALPACA" and self.stream:
            logging.info("📶 Starting Alpaca WebSocket stream…")
            # Use Alpaca stream’s internal coroutine directly
            await self.stream._run_forever()
        else:
            # fallback simulated ticks
            while True:
                await asyncio.sleep(2)
                tick = {
                    s: {
                        "price": round(100 + random.random() * 10, 2),
                        "volume": random.randint(1000, 10000),
                        "volatility": round(random.random() * 2, 3),
                        "ts": dt.datetime.utcnow().isoformat(),
                    }
                    for s in self.symbols
                }
                yield tick


    # ------------------------------------------------------------
    def latest(self):
        return pd.DataFrame.from_dict(self._latest, orient="index") if self._latest else None
