from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator
import numpy as np
import pandas as pd
from pathlib import Path

from ..utils.timebars import make_time_index_tzaware

@dataclass
class ReplayConfig:
    path: str | Path                # CSV or Parquet
    symbol: str
    tz: str = "America/New_York"
    # expected columns (L1): timestamp, bid, ask, bid_size, ask_size
    # optionally: last, last_size, micro_alpha, est_volume
    start: Optional[str] = None
    end: Optional[str] = None
    drop_na: bool = True
    # execution model params
    slip_sigma_bps: float = 2.0     # ~2 bps slip stdev baseline
    passive_fill_prob: float = 0.35 # prob of getting hit when posting
    mid_fill_prob: float = 0.65     # midpoint peg likelihood
    market_fill_prob: float = 1.00  # market orders always fill
    fill_qty_noise: float = 0.20    # noise on requested qty
    min_spread: float = 1e-4        # guardrail for zero spreads
    # latency model
    latency_ms_mean: float = 25.0
    latency_ms_std: float = 10.0

class ReplayMarketSim:
    """
    Deterministic sequence of L1/L2 snapshots with a stochastic fill model
    that depends on aggression, spread, imbalance, and simple order-flow proxies.
    - aggression: 0=passive (post), 1=mid, 2=market
    """
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)
        self.df = self._load_df(cfg)
        self._iter: Optional[Iterator[pd.Series]] = None
        self._cursor = 0
        self._current: Optional[pd.Series] = None

    # ---------- Public API ----------
    def reset(self):
        self._iter = (row for _, row in self.df.iterrows())
        self._cursor = 0
        self._current = None

    def snapshot(self) -> Dict[str, Any]:
        """Advance and return current snapshot as a dict."""
        if self._iter is None:
            self.reset()
        try:
            self._current = next(self._iter)
            self._cursor += 1
        except StopIteration:
            # loop last row to keep env alive (caller can terminate)
            self._current = self.df.iloc[-1]

        bid = float(self._current["bid"])
        ask = float(self._current["ask"])
        bid_size = int(self._current.get("bid_size", 0))
        ask_size = int(self._current.get("ask_size", 0))
        spread = max(ask - bid, self.cfg.min_spread)
        mid = (ask + bid) / 2.0
        imb = 0.0
        denom = (bid_size + ask_size)
        if denom > 0:
            imb = (bid_size - ask_size) / denom

        vol = float(self._current.get("volatility", 0.002))
        micro_alpha = float(self._current.get("micro_alpha", 0.0))
        arrival = float(self._current.get("arrival_price", mid))
        est_vol = float(self._current.get("est_volume", 1000))

        return {
            "mid": mid, "spread": spread, "imbalance": imb,
            "volatility": vol, "arrival_price": arrival,
            "latency_ms": float(self.rng.normal(self.cfg.latency_ms_mean, self.cfg.latency_ms_std)),
            "participation": min(0.95, max(0.01, est_vol / 1e6)),
            "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size,
            "micro_alpha": micro_alpha, "est_volume": est_vol
        }

    def execute(self, qty: int, aggression: int):
        """
        Return (fill_px, fill_qty, snapshot_after).
        aggression: 0=passive, 1=mid, 2=market
        """
        s = self._current if self._current is not None else self.df.iloc[self._cursor]
        bid = float(s["bid"]); ask = float(s["ask"])
        spread = max(ask - bid, self.cfg.min_spread)
        mid = (ask + bid) / 2.0

        # stochastic fill probability
        base_prob = [self.cfg.passive_fill_prob, self.cfg.mid_fill_prob, self.cfg.market_fill_prob][int(aggression)]
        # harder fills when spread is wide or imbalance is against us
        imb = 0.0
        bs, os = int(s.get("bid_size", 0)), int(s.get("ask_size", 0))
        if (bs + os) > 0:
            imb = (bs - os) / (bs + os)
        # assume buy side for cost estimate outside; we just compute fill mechanics
        against = max(0.0, -imb)  # if imbalance negative, worse for buys
        prob = base_prob * (1.0 - 0.3 * against) * (1.0 - 0.1 * math.log1p(spread / max(mid, 1e-6) * 1e4))

        did_fill = self.rng.random() < max(0.0, min(1.0, prob))
        qty_jitter = max(0.0, 1.0 + self.rng.normal(0, self.cfg.fill_qty_noise))
        fill_qty = int(min(qty, qty * qty_jitter)) if did_fill else 0

        # slip model (in bps relative to mid)
        slip_bps = float(self.rng.normal(0.0, self.cfg.slip_sigma_bps))
        slip = (slip_bps / 1e4) * mid

        if aggression == 2:           # market
            fill_px = ask + slip
        elif aggression == 1:         # mid
            fill_px = mid + 0.25 * spread + slip
        else:                         # passive (post on bid for buy)
            fill_px = bid + slip

        # evolve a step forward
        snap = self.snapshot()
        return float(fill_px), int(fill_qty), snap

    # ---------- Internals ----------
    def _load_df(self, cfg: ReplayConfig) -> pd.DataFrame:
        p = Path(cfg.path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {p}")

        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        def pick(name): return cols.get(name, name)
        # rename to canonical lowercase
        df.columns = [c.lower() for c in df.columns]
        # require minimal L1
        required = ["timestamp", "bid", "ask"]
        for r in required:
            if r not in df.columns:
                raise ValueError(f"Replay requires column: {r}")

        # timezone-safe index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = make_time_index_tzaware(df, tz=cfg.tz)

        # trim range
        if cfg.start:
            df = df.loc[pd.Timestamp(cfg.start, tz=cfg.tz):]
        if cfg.end:
            df = df.loc[:pd.Timestamp(cfg.end, tz=cfg.tz)]

        # Optional cleans
        if cfg.drop_na:
            df = df.dropna(subset=["bid", "ask"])

        # derive sizes if missing
        if "bid_size" not in df.columns: df["bid_size"] = 0
        if "ask_size" not in df.columns: df["ask_size"] = 0

        # rough realized volatility proxy over short window (for reward/cost)
        if "volatility" not in df.columns:
            px = ((df["bid"] + df["ask"]) / 2.0).astype(float)
            rv = px.pct_change().rolling(100, min_periods=10).std().fillna(method="bfill").fillna(0.002)
            df["volatility"] = rv.clip(0, 0.05)

        # arrival reference if absent
        if "arrival_price" not in df.columns:
            df["arrival_price"] = ((df["bid"] + df["ask"]) / 2.0)

        # simple volume estimate if absent
        if "est_volume" not in df.columns:
            df["est_volume"] = 1000.0

        return df
