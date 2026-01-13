import asyncio
import json
import logging
import websockets
import yfinance as yf
from typing import Dict, List

logger = logging.getLogger("DataFeedLive")


class DataFeedLive:
    """
    Streams real-time quotes from Alpaca (IEX feed) and falls back
    to yfinance when the market is closed or data is unavailable.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_ws = "wss://stream.data.alpaca.markets/v2/iex"   # free IEX feed
        self.key = cfg.get("broker", {}).get("key")
        self.secret = cfg.get("broker", {}).get("secret")
        self.cache: Dict[str, float] = {}
        self.connected = False
        self._ws_task = None
        self._symbols: List[str] = []

    async def connect(self, symbols: List[str]):
        self._symbols = symbols or []
        if not self.key or not self.secret:
            logger.warning("Missing Alpaca credentials — skipping websocket stream.")
            return

        async def _runner():
            while True:
                try:
                    async with websockets.connect(
                        self.base_ws, ping_interval=20, ping_timeout=20
                    ) as ws:
                        # authenticate and subscribe
                        await ws.send(
                            json.dumps({"action": "auth", "key": self.key, "secret": self.secret})
                        )
                        await ws.send(json.dumps({"action": "subscribe", "quotes": self._symbols}))
                        self.connected = True
                        logger.info(f"📡 Alpaca IEX stream connected for {self._symbols}")

                        async for msg in ws:
                            data = json.loads(msg)
                            for event in data:
                                if event.get("T") == "q":  # quote
                                    sym = event["S"]
                                    bp = event.get("bp")
                                    ap = event.get("ap")
                                    # midpoint or bid price as proxy
                                    price = (
                                        float(bp)
                                        if bp
                                        else float(ap)
                                        if ap
                                        else self.cache.get(sym, 0.0)
                                    )
                                    if price:
                                        self.cache[sym] = price
                except Exception as e:
                    self.connected = False
                    logger.warning(f"Stream closed: {e}")
                    await asyncio.sleep(3)  # reconnect back-off

        self._ws_task = asyncio.create_task(_runner())

    async def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Returns a dict {symbol: price}.  Prefers live cache; falls back
        to cached values or yfinance if nothing live is available.
        """
        # ✅ Prefer websocket cache
        if self.connected and all(s in self.cache for s in symbols):
            return {s: self.cache[s] for s in symbols}

        # partial cache
        partial = {s: self.cache[s] for s in symbols if s in self.cache}
        if len(partial) == len(symbols):
            return partial

        # 🧠 Fallback — yfinance when after-hours / no stream
        try:
            logger.debug("Falling back to yfinance for prices.")
            data = yf.download(
                tickers=" ".join(symbols),
                period="1d",
                interval="1m",
                progress=False,
                threads=False,
            )
            prices: Dict[str, float] = {}
            for sym in symbols:
                try:
                    prices[sym] = float(data["Close"][sym].dropna().iloc[-1])
                except Exception:
                    pass
            if prices:
                self.cache.update(prices)
                logger.info(f"📈 yfinance fallback prices fetched: {prices}")
                return prices
        except Exception as e:
            logger.warning(f"yfinance fallback failed: {e}")

        # still nothing
        return partial
