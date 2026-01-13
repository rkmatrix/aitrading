# ai/reward/vol_aware_fusion.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from .volatility_metrics import realized_vol, ewma_vol, max_drawdown

# ----- Config dataclasses -----

@dataclass
class AssetVolSpec:
    name: str
    vol_metric: str = "ewma"         # "realized" | "ewma"
    vol_span: int = 20               # ewma span (if used)
    returns_window: Optional[int] = None  # override global
    weight: float = 1.0              # aggregation weight

@dataclass
class VolAwareConfig:
    # Global windows
    returns_window: int = 60
    equity_window: int = 300

    # Single-metric defaults (used if no per-asset override)
    vol_metric: str = "ewma"          # "realized" | "ewma"
    vol_span: int = 20

    # Per-asset configuration (optional)
    assets: List[AssetVolSpec] = field(default_factory=list)
    aggregate_mode: str = "weighted_mean"  # "weighted_mean" | "max" | "mean"

    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "vol_low": 0.004,
        "vol_high": 0.02
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "pnl": 1.0,
        "vol_penalty": 1.0,
        "dd_penalty": 0.5,
        "trade_cost": 1.0
    })
    risk_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "base": 1.0,
        "low_vol": 1.1,
        "high_vol": 0.6
    })
    penalty_scales: Dict[str, float] = field(default_factory=lambda: {
        "vol": 1.0,
        "dd": 1.0
    })
    clamp_reward: Optional[float] = 10.0

# ----- Helper functions -----

def _compute_series_vol(returns: List[float], *, metric: str, span: int, window: int) -> float:
    if metric == "realized":
        return realized_vol(returns, window=window)
    return ewma_vol(returns, span=span)

def _aggregate_vol(per_asset_vols: Dict[str, Tuple[float, float]], mode: str) -> float:
    """
    per_asset_vols: name -> (vol, weight)
    mode:
      - weighted_mean: sum(vol*w)/sum(w)
      - mean: average of vol (ignores weights)
      - max: maximum vol across assets
    """
    if not per_asset_vols:
        return 0.0
    vs = [(v, w) for (_, (v, w)) in per_asset_vols.items()]
    if mode == "max":
        return float(max(v for v, _ in vs))
    if mode == "mean":
        return float(np.mean([v for v, _ in vs]))
    # default weighted mean
    num = sum(v * (w if w > 0 else 0.0) for v, w in vs)
    den = sum((w if w > 0 else 0.0) for _, w in vs)
    if den <= 1e-12:
        return float(np.mean([v for v, _ in vs]))
    return float(num / den)

# ----- Fuser -----

class VolAwareRewardFuser:
    """
    Supports both single-stream and per-asset returns:
      - Single: supply 'ret' (float) in info. We'll update a global returns buffer.
      - Multi: supply 'asset_returns' = { "AAPL": 0.0012, "MSFT": -0.0005, ... }.
               We'll track per-asset buffers and compute per-asset vols using the asset list in config.
    Equity path is still tracked globally for drawdown.
    """
    def __init__(self, cfg: VolAwareConfig, logger=None):
        self.cfg = cfg
        self.logger = logger  # optional RewardLogger shim
        self._returns: List[float] = []             # global
        self._equity: List[float] = []
        self._asset_returns: Dict[str, List[float]] = {}  # per-asset name -> list[float]
        # quick lookup for asset specs
        self._asset_map: Dict[str, AssetVolSpec] = {a.name: a for a in (cfg.assets or [])}

    def reset(self):
        self._returns.clear()
        self._equity.clear()
        self._asset_returns.clear()

    # ----- internal computations -----

    def _compute_vol_single(self) -> float:
        # Use global settings
        if self.cfg.vol_metric == "realized":
            return realized_vol(self._returns, window=self.cfg.returns_window)
        return ewma_vol(self._returns, span=self.cfg.vol_span)

    def _compute_vol_multi(self) -> Tuple[float, Dict[str, float]]:
        per_asset_vols: Dict[str, Tuple[float, float]] = {}
        per_asset_scalar: Dict[str, float] = {}
        for name, buf in self._asset_returns.items():
            spec = self._asset_map.get(name)
            if spec is None:
                # fallback to global settings
                metric = self.cfg.vol_metric
                span = self.cfg.vol_span
                window = self.cfg.returns_window
                weight = 1.0
            else:
                metric = spec.vol_metric or self.cfg.vol_metric
                span = spec.vol_span if spec.vol_metric == "ewma" else self.cfg.vol_span
                window = spec.returns_window or self.cfg.returns_window
                weight = spec.weight if spec.weight > 0 else 1.0
            vol = _compute_series_vol(buf, metric=metric, span=span, window=window)
            per_asset_vols[name] = (vol, weight)
            per_asset_scalar[name] = vol
        agg = _aggregate_vol(per_asset_vols, mode=self.cfg.aggregate_mode)
        return agg, per_asset_scalar

    def _risk_multiplier(self, vol: float) -> float:
        low = self.cfg.thresholds.get("vol_low", 0.004)
        high = self.cfg.thresholds.get("vol_high", 0.02)
        if vol <= low:
            return self.cfg.risk_multipliers.get("low_vol", 1.1)
        if vol >= high:
            return self.cfg.risk_multipliers.get("high_vol", 0.6)
        return self.cfg.risk_multipliers.get("base", 1.0)

    def _update_buffers(self, info: Dict[str, Any]):
        # Per-asset returns (preferred when present)
        aset = info.get("asset_returns")
        if isinstance(aset, dict) and aset:
            for name, r in aset.items():
                buf = self._asset_returns.setdefault(name, [])
                try:
                    buf.append(float(r))
                except Exception:
                    buf.append(0.0)
                # trim
                maxw = max(self.cfg.returns_window, *(a.returns_window or self.cfg.returns_window for a in self.cfg.assets)) if self.cfg.assets else self.cfg.returns_window
                if len(buf) > maxw * 2:
                    self._asset_returns[name] = buf[-(maxw * 2):]
            # Also produce a global proxy return as weighted mean for legacy consumers
            try:
                weights = []
                rets = []
                for name, r in aset.items():
                    w = self._asset_map.get(name).weight if name in self._asset_map else 1.0
                    weights.append(max(w, 0.0))
                    rets.append(float(r))
                wsum = sum(weights) or 1.0
                proxy = sum(w * r for w, r in zip(weights, rets)) / wsum
            except Exception:
                proxy = 0.0
            self._returns.append(float(proxy))
        else:
            # Single return path
            ret = info.get("ret")
            if ret is None:
                if "equity" in info and self._equity:
                    prev = self._equity[-1]
                    now = float(info["equity"])
                    if prev != 0:
                        ret = (now - prev) / abs(prev)
                elif "pnl" in info and "equity" in info and info["equity"]:
                    ret = float(info["pnl"]) / max(abs(float(info["equity"])), 1e-9)
                else:
                    ret = 0.0
            self._returns.append(float(ret))

        # Trim global buffer
        if len(self._returns) > max(self.cfg.returns_window, self.cfg.equity_window) * 2:
            self._returns = self._returns[-(max(self.cfg.returns_window, self.cfg.equity_window) * 2):]

        # Equity path
        if "equity" in info:
            self._equity.append(float(info["equity"]))
            if len(self._equity) > self.cfg.equity_window * 2:
                self._equity = self._equity[-(self.cfg.equity_window * 2):]

    # ----- public API -----

    def step(self, info: Dict[str, Any], *, step_index: Optional[int] = None) -> Dict[str, float]:
        self._update_buffers(info)

        # Compute volatility (single or multi)
        if self._asset_returns:
            vol, per_asset_vol = self._compute_vol_multi()
        else:
            vol = self._compute_vol_single()
            per_asset_vol = {}

        # Drawdown on global equity curve
        dd = max_drawdown(self._equity[-self.cfg.equity_window:]) if self._equity else 0.0  # negative
        pnl = float(info.get("pnl", 0.0))
        cost = float(info.get("trade_cost", 0.0))

        # Penalties (positive magnitudes)
        vol_pen = self.cfg.penalty_scales["vol"] * float(vol)
        dd_pen = self.cfg.penalty_scales["dd"] * abs(float(dd))

        # Linear fusion
        w = self.cfg.weights
        linear = (w["pnl"] * pnl) - (w["vol_penalty"] * vol_pen) - (w["dd_penalty"] * dd_pen) - (w["trade_cost"] * cost)

        # Risk multiplier
        rmult = self._risk_multiplier(vol)
        total = rmult * linear

        # Clamp
        c = self.cfg.clamp_reward
        if c is not None:
            total = float(np.clip(total, -abs(c), abs(c)))

        out = {
            "reward": float(total),
            "pnl": pnl,
            "trade_cost": cost,
            "vol": float(vol),
            "dd": float(dd),
            "vol_pen": float(vol_pen),
            "dd_pen": float(dd_pen),
            "risk_mult": float(rmult),
            "linear": float(linear),
        }
        # Expand per-asset vol into namespaced keys for logging (optional)
        for k, v in per_asset_vol.items():
            out[f"vol.asset.{k}"] = float(v)

        # Optional logger
        if self.logger is not None and step_index is not None:
            try:
                self.logger.log(step_index, out)
            except Exception:
                pass

        return out
