# ai/allocators/adaptive_weights.py
"""
Adaptive, online signal-weighting for PPO / Momentum / MeanRev.

Core idea:
- Maintain per-symbol, per-strategy exponential moving stats of returns.
- Turn those stats into adaptive weights α, β, γ such that α+β+γ=1.
- Persist weights+state to disk and evolve them over time.

How to use:
    aw = AdaptiveWeights(store_path="data/adaptive_weights.json")
    w = aw.get_weights("AAPL")  # {'ppo': 0.34, 'momentum': 0.33, 'meanrev': 0.33}
    # ... later after realizing a bar's returns per strategy ...
    aw.update("AAPL", returns={'ppo': 0.002, 'momentum': -0.001, 'meanrev': 0.0005})
    # weights auto-adjust using EMA of performance and a Sharpe-like proxy
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# --------- Small numerics helpers ---------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


# --------- EMA tracker with variance (for Sharpe-like metric) ---------
@dataclass
class EmaStats:
    """Tracks EMA mean and variance for returns stream."""
    alpha: float = 0.10  # smoothing factor for mean
    beta: float = 0.10   # smoothing for variance of returns around the EMA mean
    mean: float = 0.0
    var: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> None:
        if not self.initialized:
            # Bootstrap with first observation
            self.mean = x
            self.var = 0.0
            self.initialized = True
            return

        # EMA mean
        prev_mean = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x

        # EMA variance around EMA mean (Welford-like but EMA)
        # Using beta separate from alpha allows independent smoothing
        delta = x - prev_mean
        delta2 = x - self.mean
        incr_var = delta * delta2  # equivalent to (x-mean_prev)*(x-mean_new)
        self.var = (1 - self.beta) * self.var + self.beta * incr_var

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def sharpe_like(self, eps: float = 1e-9) -> float:
        """Return a stabilized Sharpe proxy."""
        return _safe_div(self.mean, self.std + eps, default=0.0)


# --------- Per-strategy state ---------
@dataclass
class StrategyPerf:
    ema_ret: EmaStats = field(default_factory=lambda: EmaStats(alpha=0.10, beta=0.10))
    # Optional: track hit-rate EMA too (not used directly; can be used later)
    hits: float = 0.0
    total: float = 0.0

    def update(self, r: float) -> None:
        self.ema_ret.update(r)
        self.total += 1.0
        if r > 0:
            self.hits += 1.0

    @property
    def hit_rate(self) -> float:
        return _safe_div(self.hits, self.total, default=0.0)


# --------- Symbol state ---------
@dataclass
class SymbolState:
    strategies: Dict[str, StrategyPerf] = field(default_factory=lambda: {
        "ppo": StrategyPerf(),
        "momentum": StrategyPerf(),
        "meanrev": StrategyPerf(),
    })
    # The last computed, normalized weights (sum to 1)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "ppo": 1/3, "momentum": 1/3, "meanrev": 1/3
    })
    # Optional: count of updates for warmup control
    updates: int = 0


# --------- Main class ---------
class AdaptiveWeights:
    """
    Online meta-allocator producing α_t, β_t, γ_t based on recent performance.
    Combines two components per strategy:
        score = w_ret * EMA(Return) + w_sharpe * Sharpe_like(EMA)
    Then maps scores -> positive via exp and normalizes to 1.
    """

    def __init__(
        self,
        store_path: str = "data/adaptive_weights.json",
        w_ret: float = 0.50,
        w_sharpe: float = 0.50,
        softmax_temp: float = 1.0,
        min_weight: float = 0.05,
        warmup_steps: int = 20,
    ):
        self.store_path = store_path
        self.w_ret = w_ret
        self.w_sharpe = w_sharpe
        self.softmax_temp = max(1e-6, softmax_temp)
        self.min_weight = min_weight  # floor per strategy
        self.warmup_steps = warmup_steps
        self._state: Dict[str, SymbolState] = {}
        self._load()

    # ---------- Persistence ----------
    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Rehydrate EMA objects
            for sym, blob in raw.items():
                sym_state = SymbolState()
                sym_state.updates = int(blob.get("updates", 0))
                sym_state.weights = blob.get("weights", {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3})
                for k in ["ppo", "momentum", "meanrev"]:
                    sp_raw = blob.get("strategies", {}).get(k, {})
                    ema = sp_raw.get("ema_ret", {})
                    perf = StrategyPerf()
                    perf.ema_ret = EmaStats(
                        alpha=ema.get("alpha", 0.10),
                        beta=ema.get("beta", 0.10),
                        mean=ema.get("mean", 0.0),
                        var=ema.get("var", 0.0),
                        initialized=ema.get("initialized", False),
                    )
                    perf.hits = float(sp_raw.get("hits", 0.0))
                    perf.total = float(sp_raw.get("total", 0.0))
                    sym_state.strategies[k] = perf
                self._state[sym] = sym_state
        except Exception:
            # Corrupt or incompatible file → ignore and start fresh
            self._state = {}

    def _save(self) -> None:
        out = {}
        for sym, sym_state in self._state.items():
            s_blob = {
                "updates": sym_state.updates,
                "weights": sym_state.weights,
                "strategies": {},
            }
            for k, perf in sym_state.strategies.items():
                s_blob["strategies"][k] = {
                    "ema_ret": {
                        "alpha": perf.ema_ret.alpha,
                        "beta": perf.ema_ret.beta,
                        "mean": perf.ema_ret.mean,
                        "var": perf.ema_ret.var,
                        "initialized": perf.ema_ret.initialized,
                    },
                    "hits": perf.hits,
                    "total": perf.total,
                }
            out[sym] = s_blob

        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # ---------- Public API ----------
    def get_weights(self, symbol: str) -> Dict[str, float]:
        """Return current weights for symbol; if unseen, equal weights."""
        st = self._state.get(symbol)
        if st is None:
            st = SymbolState()
            self._state[symbol] = st
            self._save()
        return dict(st.weights)

    def update(self, symbol: str, returns: Dict[str, float], autosave: bool = True) -> Dict[str, float]:
        """
        Update performance stats for symbol and recompute weights.

        Args:
            returns: e.g., {'ppo': 0.002, 'momentum': -0.001, 'meanrev': 0.0005}
                     Values should be realized strategy-attributed returns for the bar/day.
                     If attribution is approximate, it's still useful—noise gets smoothed by EMA.
        """
        st = self._state.get(symbol)
        if st is None:
            st = SymbolState()
            self._state[symbol] = st

        # 1) Update EMAs
        for k in ["ppo", "momentum", "meanrev"]:
            r = float(returns.get(k, 0.0))
            st.strategies[k].update(r)

        st.updates += 1

        # 2) Compute composite score per strategy
        scores: Dict[str, float] = {}
        for k, perf in st.strategies.items():
            ema_r = perf.ema_ret.mean
            sharpe_like = perf.ema_ret.sharpe_like()
            score = self.w_ret * ema_r + self.w_sharpe * sharpe_like
            scores[k] = score

        # 3) Softmax → positive weights
        #    During warmup, blend toward equal weights to avoid early overfitting.
        w = self._softmax(scores, temp=self.softmax_temp)

        if st.updates < self.warmup_steps:
            # Linear blend toward equal weights for warmup
            t = st.updates / float(max(1, self.warmup_steps))
            eq = {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3}
            w = {k: (1 - t) * eq[k] + t * w[k] for k in w.keys()}

        # 4) Apply per-strategy floor and renormalize
        w = {k: _clip(v, self.min_weight, 1.0) for k, v in w.items()}
        s = sum(w.values())
        if s <= 0:
            w = {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3}
        else:
            w = {k: v / s for k, v in w.items()}

        st.weights = w

        if autosave:
            self._save()

        return dict(w)

    # ---------- Internals ----------
    @staticmethod
    def _softmax(scores: Dict[str, float], temp: float = 1.0) -> Dict[str, float]:
        # numeric stability: subtract max
        mx = max(scores.values()) if scores else 0.0
        exps = {k: math.exp((v - mx) / max(1e-6, temp)) for k, v in scores.items()}
        z = sum(exps.values())
        if z <= 0:
            n = len(scores) if scores else 3
            return {k: 1.0 / n for k in scores.keys()}
        return {k: v / z for k, v in exps.items()}
