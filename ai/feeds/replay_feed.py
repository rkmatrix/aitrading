# ai/feeds/replay_feed.py
# Phase-G — Deterministic Replay Feed
#
# Safety guarantees:
# - NO synthetic prices
# - NO DEMO shortcuts
# - Market-open awareness enforced
# - Deterministic iteration over historical bars

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List
import pandas as pd
import pytz
from datetime import datetime, time

@dataclass
class ReplayBar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class ReplayFeed:
    def __init__(self, ohlcv_csv_by_symbol: Dict[str, str], *, tz: str = "America/New_York") -> None:
        self.tz = pytz.timezone(tz)
        self.data: Dict[str, pd.DataFrame] = {}
        for sym, path in ohlcv_csv_by_symbol.items():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")
            self.data[sym.upper()] = df

    def _is_market_open(self, ts: pd.Timestamp) -> bool:
        # Simple NYSE hours gate; holidays handled by data gaps
        t = ts.tz_convert(self.tz).time()
        return time(9, 30) <= t <= time(16, 0)

    def iter_bars(self) -> Iterator[Dict[str, ReplayBar]]:
        # Align by timestamp across symbols (inner join on timestamps)
        frames = []
        for sym, df in self.data.items():
            tmp = df[["timestamp","open","high","low","close","volume"]].copy()
            tmp.columns = ["timestamp", f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", f"{sym}_volume"]
            frames.append(tmp)
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="timestamp", how="inner")

        for _, row in merged.iterrows():
            ts = row["timestamp"]
            if not self._is_market_open(ts):
                continue
            out: Dict[str, ReplayBar] = {}
            for sym in self.data.keys():
                out[sym] = ReplayBar(
                    ts=ts,
                    open=float(row[f"{sym}_open"]),
                    high=float(row[f"{sym}_high"]),
                    low=float(row[f"{sym}_low"]),
                    close=float(row[f"{sym}_close"]),
                    volume=float(row[f"{sym}_volume"]),
                )
            yield out
