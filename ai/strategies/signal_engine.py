"""
ai/strategies/signal_engine.py
Phase 92.3 – Multi-Factor SignalEngine
--------------------------------------

Enhancements:
    ✔ config_path support
    ✔ cfg (inline config) support
    ✔ integrates PortfolioBrain trend + realized vol
    ✔ safe fallbacks if PortfolioBrain unavailable
    ✔ ML variance placeholder (FusionEngine Phase92 expects it)
    ✔ does NOT break your existing momentum/meanrev/macro logic
    ✔ fully backward compatible with older phases

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import yaml
from utils.logger import log

# For compatibility with PortfolioBrain (Phase-92)
try:
    from ai.allocators.portfolio_brain import PortfolioBrain
except Exception:
    PortfolioBrain = None


# -----------------------------------------------------------
# Helper to load YAML configs
# -----------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log(f"⚠️ SignalEngine config load failed ({path}): {e}")
        return {}


# ======================================================================
# Phase-92.3 SignalEngine
# ======================================================================
class SignalEngine:
    """
    Multi-signal engine (momentum, mean-rev, macro) + Phase-92 context features.

    This engine now produces a dict:
        {
            "momentum": Series,
            "meanrev": Series,
            "macro": Series,
            "trend_strength": Series,
            "volatility": Series,
            "ml_variance": float,
        }

    This matches exactly what the Phase-92 FusionEngine expects.
    """

    def __init__(
        self,
        symbols: list[str],
        momentum_window: int = 20,
        meanrev_window: int = 5,
        config_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        portfolio_brain: Optional[Any] = None,
    ):
        self.symbols = symbols
        self.momentum_window = momentum_window
        self.meanrev_window = meanrev_window

        # Load config from file or dict
        self.cfg = cfg or _load_yaml(config_path)

        # PortfolioBrain is optional but used for Phase-92 trend/vol context
        self.pbrain = portfolio_brain

        log(f"📡 SignalEngine initialized for {len(symbols)} symbols (Phase 92.3)")

    # -----------------------------------------------------------
    # Momentum
    # -----------------------------------------------------------
    def momentum(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(0.0, index=self.symbols)
        rets = prices.pct_change()
        mom = (
            rets.rolling(self.momentum_window).mean() /
            (rets.rolling(self.momentum_window).std() + 1e-8)
        )
        latest = mom.iloc[-1].fillna(0.0)
        latest.name = "momentum"
        return latest

    # -----------------------------------------------------------
    # Mean-Reversion
    # -----------------------------------------------------------
    def mean_reversion(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(0.0, index=self.symbols)
        short_ma = prices.rolling(self.meanrev_window).mean()
        z = (prices - short_ma) / (prices.rolling(self.meanrev_window).std() + 1e-8)
        mr = -z.iloc[-1].fillna(0.0)
        mr.name = "meanrev"
        return mr

    # -----------------------------------------------------------
    # Macro (volatility-based)
    # -----------------------------------------------------------
    def macro_features(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(0.0, index=self.symbols)
        vol = prices.pct_change().rolling(30).std().iloc[-1]
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        macro = -vol.fillna(0.0)
        macro.name = "macro"
        return macro

    # -----------------------------------------------------------
    # NEW – Trend Strength (Phase-92)
    # -----------------------------------------------------------
    def trend_strength(self) -> pd.Series:
        if not self.pbrain:
            # fallback: no trend → zero
            return pd.Series(0.0, index=self.symbols, name="trend_strength")

        vals = {}
        for s in self.symbols:
            try:
                vals[s] = self.pbrain.get_trend_strength(s)
            except Exception:
                vals[s] = 0.0

        return pd.Series(vals, name="trend_strength")

    # -----------------------------------------------------------
    # NEW – Realized Volatility (Phase-92)
    # -----------------------------------------------------------
    def realized_volatility(self) -> pd.Series:
        if not self.pbrain:
            return pd.Series(1.0, index=self.symbols, name="volatility")

        vals = {}
        for s in self.symbols:
            try:
                v = self.pbrain.get_realized_vol(s)
                vals[s] = float(np.clip(v, 0.2, 5.0))   # clamp extreme
            except Exception:
                vals[s] = 1.0

        return pd.Series(vals, name="volatility")

    # -----------------------------------------------------------
    # NEW – ML variance placeholder (Phase-92)
    #   FusionEngine expects a single float, not per-symbol.
    # -----------------------------------------------------------
    def ml_variance(self) -> float:
        # Future version: hook in AlphaZoo variance estimates
        return 0.001  # stable low variance as placeholder

    # -----------------------------------------------------------
    # All Signals (Phase-92 multi-agent compatibility)
    # -----------------------------------------------------------
    def all_signals(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a dict including both classical signals and
        Phase-92 context features.
        """

        signals = {
            "momentum": self.momentum(prices),
            "meanrev": self.mean_reversion(prices),
            "macro": self.macro_features(prices),
            "trend_strength": self.trend_strength(),
            "volatility": self.realized_volatility(),
            "ml_variance": self.ml_variance(),
        }

        return signals
