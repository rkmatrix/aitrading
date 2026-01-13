import asyncio
import csv
import glob
import logging
import os
from typing import AsyncIterator, Dict, Any, List

log = logging.getLogger(__name__)

# choose Alpaca stream if available
_ALPACA_DATA_AVAILABLE = True
try:
    from alpaca.data.live import StockDataStream
except Exception:
    _ALPACA_DATA_AVAILABLE = False
    log.warning("alpaca-py data stream not available; will use CSV in DRY_RUN.")

class Tick:
    __slots__ = ("symbol", "ts", "bid", "ask", "mid")
    def __init__(self, symbol: str, ts: float, bid: float, ask: float):
        self.symbol = symbol
        self.ts = ts
        self.bid = bid
        self.ask = ask
        self.mid = (bid + ask)/2.0 if (bid and ask) else bid or ask

class BaseTickStream:
    async def stream(self) -> AsyncIterator[Tick]:
        raise NotImplementedError

class CSVTickStream(BaseTickStream):
    def __init__(self, csv_glob: str, symbols: List[str]):
        self.csv_glob = csv_glob
        self.symbols = set(symbols)

    async def stream(self) -> AsyncIterator[Tick]:
        files = sorted(glob.glob(self.csv_glob))
        for fp in files:
            with open(fp, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    sym = row.get("symbol") or row.get("Symbol") or row.get("SYM")
                    if sym not in self.symbols:
                        continue
                    ts = float(row.get("ts") or row.get("timestamp") or row.get("time") or 0)
                    bid = float(row.get("bid") or row.get("Bid") or row.get("BID") or row.get("price") or 0)
                    ask = float(row.get("ask") or row.get("Ask") or row.get("ASK") or row.get("price") or 0)
                    yield Tick(sym, ts, bid, ask)
                    await asyncio.sleep(0.01)  # slow down for demo

class AlpacaTickStream(BaseTickStream):
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        if not _ALPACA_DATA_AVAILABLE:
            raise RuntimeError("alpaca-py data not installed.")
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _run(self):
        stream = StockDataStream(self.api_key, self.api_secret)
        async def on_quote(q):
            bid = float(q.bid_price) if q.bid_price else 0.0
            ask = float(q.ask_price) if q.ask_price else 0.0
            ts = float(q.timestamp.timestamp())
            await self._queue.put(Tick(q.symbol, ts, bid, ask))

        for s in self.symbols:
            stream.subscribe_quotes(on_quote, s)

        await stream.run()

    async def stream(self) -> AsyncIterator[Tick]:
        task = asyncio.create_task(self._run())
        try:
            while True:
                tick = await self._queue.get()
                yield tick
        finally:
            task.cancel()
