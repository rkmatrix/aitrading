"""
signals.py – Phase 28 Stable Reward Signal Definitions
Each signal now self-heals internal state to avoid KeyErrors.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


# ---------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------
class RewardSignal:
    """Base class for reward components."""

    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self._state: Dict[str, Any] = {}

    def ensure(self, key: str, default):
        """Ensure state key exists."""
        if key not in self._state:
            self._state[key] = default
        return self._state[key]

    def reset(self):
        self._state.clear()

    def step(self, info: Dict[str, Any]) -> float:
        raise NotImplementedError


@dataclass
class SignalOutput:
    name: str
    value: float


# ---------------------------------------------------------------------
# PnL Signal
# ---------------------------------------------------------------------
class PnLSignal(RewardSignal):
    name = "pnl"

    def __init__(self, params=None):
        super().__init__(params)
        self.ensure("total_pnl", 0.0)

    def reset(self):
        self._state["total_pnl"] = 0.0

    def step(self, info: Dict[str, Any]) -> float:
        scale = float(self.params.get("scale", 1.0))
        v = float(info.get("step_pnl", 0.0)) * scale
        self._state["total_pnl"] += v
        return float(np.tanh(v))


# ---------------------------------------------------------------------
# Risk Signal
# ---------------------------------------------------------------------
class RiskSignal(RewardSignal):
    name = "risk"

    def __init__(self, params=None):
        super().__init__(params)
        self.ensure("ret_hist", [])

    def reset(self):
        self._state["ret_hist"] = []

    def step(self, info: Dict[str, Any]) -> float:
        hist = self.ensure("ret_hist", [])
        target_vol = float(self.params.get("target_vol", 0.02))
        vol_window = int(self.params.get("vol_window", 30))
        r = float(info.get("step_return", 0.0))
        hist.append(r)
        if len(hist) > vol_window:
            hist.pop(0)
        vol = float(np.std(hist)) if hist else 0.0
        if vol == 0:
            return 0.0
        return float(np.clip((target_vol - vol) / max(target_vol, 1e-6), -1.0, 1.0))


# ---------------------------------------------------------------------
# Slippage Signal
# ---------------------------------------------------------------------
class SlippageSignal(RewardSignal):
    name = "slippage"

    def __init__(self, params=None):
        super().__init__(params)
        self.ensure("count", 0)

    def reset(self):
        self._state["count"] = 0

    def step(self, info: Dict[str, Any]) -> float:
        self._state["count"] += 1
        slip_bps = float(info.get("step_slippage_bps", 0.0))
        scale = float(self.params.get("scale", 1.0))
        return float(np.clip(-slip_bps / 25.0 * scale, -1.0, 1.0))


# ---------------------------------------------------------------------
# Drawdown Signal
# ---------------------------------------------------------------------
class DrawdownSignal(RewardSignal):
    name = "drawdown"

    def __init__(self, params=None):
        super().__init__(params)
        self.ensure("equity_curve", [])
        self.ensure("peak", None)

    def reset(self):
        self._state["equity_curve"] = []
        self._state["peak"] = None

    def step(self, info: Dict[str, Any]) -> float:
        curve = self.ensure("equity_curve", [])
        peak = self.ensure("peak", None)
        eq = float(info.get("equity", 0.0))
        curve.append(eq)
        if peak is None or eq > peak:
            peak = eq
            self._state["peak"] = peak
        dd = 0.0 if peak in (None, 0.0) else (peak - eq) / max(peak, 1e-9)
        return float(np.clip(-dd, -1.0, 0.0))


# ---------------------------------------------------------------------
# Hit Rate Signal
# ---------------------------------------------------------------------
class HitRateSignal(RewardSignal):
    name = "hitrate"

    def __init__(self, params=None):
        super().__init__(params)
        self.window = int(self.params.get("window", 50))
        self.ensure("last_outcomes", [])

    def reset(self):
        self._state["last_outcomes"] = []

    def step(self, info: Dict[str, Any]) -> float:
        lo = self.ensure("last_outcomes", [])
        outcome = info.get("trade_outcome", None)
        if outcome is not None:
            try:
                lo.append(int(outcome))
            except Exception:
                pass
            if len(lo) > self.window:
                lo.pop(0)
        if not lo:
            return 0.0
        wins = sum(1 for x in lo if x > 0)
        hr = wins / len(lo)
        return float(np.clip((hr - 0.5) * 2.0, -1.0, 1.0))


# ---------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------
def build_signals(cfg: Dict[str, Any]) -> Dict[str, RewardSignal]:
    mapping = {
        "pnl": PnLSignal,
        "risk": RiskSignal,
        "slippage": SlippageSignal,
        "drawdown": DrawdownSignal,
        "hitrate": HitRateSignal,
    }
    out = {}
    for name, scfg in cfg.items():
        if not scfg.get("enabled", True):
            continue
        cls = mapping.get(name)
        if cls is None:
            continue
        out[name] = cls(params=scfg.get("params", {}))
    return out
