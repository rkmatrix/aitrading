from __future__ import annotations
import time, random, logging, math, csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class Bar:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    """Abstract feed."""
    def next_bar(self, symbol: str) -> Bar:
        raise NotImplementedError


class RandomFeed(DataFeed):
    """Cheap simulated feed; good enough for local testing."""
    def __init__(self, base_prices: Dict[str, float], interval_seconds: int = 2):
        self.prices = {s: float(p) for s, p in base_prices.items()}
        self.interval = interval_seconds

    def next_bar(self, symbol: str) -> Bar:
        p = self.prices.get(symbol, 100.0)
        # small random walk
        drift = random.uniform(-0.003, 0.003)
        new = max(0.5, p * (1.0 + drift))
        vol = abs((new - p) / p)
        volume = 1000.0 + 5000.0 * vol + random.uniform(0, 500)
        o = p
        c = new
        h = max(o, c) * (1.0 + random.uniform(0, 0.001))
        l = min(o, c) * (1.0 - random.uniform(0, 0.001))
        self.prices[symbol] = new
        time.sleep(self.interval)
        return Bar(time.time(), o, h, l, c, volume)


class CSVFeed(DataFeed):
    """
    Very simple CSV reader.
    Accepts a 'wide' CSV: ts,symbol,open,high,low,close,volume (unordered OK).
    """
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"CSV feed path not found: {self.path}")
        self.rows_by_symbol: Dict[str, List[Bar]] = {}
        with open(self.path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                sym = r["symbol"].strip()
                bar = Bar(
                    ts=float(r.get("ts") or r.get("timestamp") or time.time()),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("volume") or r.get("vol") or 0),
                )
                self.rows_by_symbol.setdefault(sym, []).append(bar)
        # progress cursors
        self.idx: Dict[str, int] = {s: 0 for s in self.rows_by_symbol}

    def next_bar(self, symbol: str) -> Bar:
        rows = self.rows_by_symbol.get(symbol, [])
        if not rows:
            # fallback to a static bar if symbol missing in CSV
            return Bar(time.time(), 100, 101, 99, 100, 1000)
        i = self.idx[symbol]
        if i >= len(rows):
            # loop
            i = 0
        bar = rows[i]
        self.idx[symbol] = i + 1
        # no sleeping; caller can throttle
        return bar


class StateObserver:
    """
    Maintains rolling features per symbol:
      - priceDelta (close vs SMA(delta_window))
      - volatility (stddev of returns over vol_window)
      - volumeRatio (volume / avgVolume(vol_avg_window))
      - rsi (RSI over rsi_period)
    """
    def __init__(
        self,
        symbols: List[str],
        feed: DataFeed,
        *,
        rsi_period: int = 14,
        vol_window: int = 20,
        delta_window: int = 5,
        vol_avg_window: int = 10,
        snapshot_csv: Optional[str] = None,
    ):
        self.symbols = symbols
        self.feed = feed
        self.rsi_period = rsi_period
        self.vol_window = vol_window
        self.delta_window = delta_window
        self.vol_avg_window = vol_avg_window
        self.snap_path = Path(snapshot_csv) if snapshot_csv else None
        if self.snap_path:
            self.snap_path.parent.mkdir(parents=True, exist_ok=True)
        # rolling storages
        self.closes: Dict[str, deque] = {s: deque(maxlen=max(self.delta_window, self.rsi_period, self.vol_window)) for s in symbols}
        self.returns: Dict[str, deque] = {s: deque(maxlen=self.vol_window) for s in symbols}
        self.volumes: Dict[str, deque] = {s: deque(maxlen=self.vol_avg_window) for s in symbols}
        self.last_state: Dict[str, Dict] = {s: {} for s in symbols}

    # --------------- math helpers ---------------
    @staticmethod
    def _sma(vals: deque) -> float:
        return sum(vals) / max(1, len(vals))

    @staticmethod
    def _std(vals: deque) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
        return math.sqrt(var)

    def _rsi(self, vals: deque) -> float:
        if len(vals) < self.rsi_period + 1:
            return 50.0
        gains, losses = 0.0, 0.0
        for i in range(-self.rsi_period, 0):
            diff = vals[i] - vals[i - 1]
            if diff >= 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / self.rsi_period
        avg_loss = losses / self.rsi_period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # --------------- core ---------------
    def step(self) -> Dict[str, Dict]:
        """
        Pulls one bar per symbol from feed, updates rolling windows,
        returns latest feature dicts keyed by symbol.
        """
        states: Dict[str, Dict] = {}
        for s in self.symbols:
            bar = self.feed.next_bar(s)
            # update rolls
            self.volumes[s].append(bar.volume)
            closes = self.closes[s]
            closes.append(bar.close)
            if len(closes) >= 2:
                ret = (closes[-1] - closes[-2]) / (closes[-2] or 1.0)
                self.returns[s].append(ret)
            # features
            sma = self._sma(deque(list(closes)[-self.delta_window:]))
            price_delta = 0.0 if sma == 0 else (bar.close - sma) / sma
            vol = self._std(self.returns[s])
            avg_vol = self._sma(self.volumes[s])
            vol_ratio = 0.0 if avg_vol == 0 else bar.volume / avg_vol
            rsi = self._rsi(closes)
            state = {
                "ts": bar.ts,
                "symbol": s,
                "close": bar.close,
                "priceDelta": price_delta,
                "volatility": vol,
                "volumeRatio": vol_ratio,
                "rsi": rsi,
            }
            states[s] = state
            self.last_state[s] = state
            if self.snap_path:
                self._write_snapshot(state)
        return states

    def get_state(self, symbol: str) -> Dict:
        return self.last_state.get(symbol) or {}

    def _write_snapshot(self, st: Dict):
        hdr = ["ts", "symbol", "close", "priceDelta", "volatility", "volumeRatio", "rsi"]
        write_header = not self.snap_path.exists()
        with open(self.snap_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(hdr)
            w.writerow([st.get(k, "") for k in hdr])
