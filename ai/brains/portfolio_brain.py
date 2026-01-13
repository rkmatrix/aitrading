# portfolio_brain.py
"""
PortfolioBrain with Adaptive Signal Weighting (Phase 8.1 integration)

Key additions:
- Import & initialize AdaptiveWeights
- Replace static blend with adaptive blend per symbol
- Provide a hook `record_strategy_outcomes(...)` to feed realized returns
  back into the adaptive engine (can be called post-bar, post-trade, or daily)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

# --- NEW: import the adaptive weighting module ---
from ai.allocators.adaptive_weights import AdaptiveWeights


class PortfolioBrain:
    def __init__(
        self,
        symbols,
        risk_free_daily: float = 0.0,
        weights_store: str = "data/adaptive_weights.json",
        enable_adaptive_weights: bool = True,
    ):
        self.symbols = list(symbols)
        self.risk_free_daily = risk_free_daily
        self.enable_adaptive_weights = enable_adaptive_weights

        # Initialize AdaptiveWeights engine (persisted across runs)
        self.aw = AdaptiveWeights(store_path=weights_store)

        # Internal caches (optional)
        self._last_weights: Dict[str, Dict[str, float]] = {}

    # -----------------------------
    # Signal ingestion (external)
    # -----------------------------
    def get_signals_for_symbol(self, symbol: str) -> Dict[str, float]:
        """
        Placeholder that your runner should fill with actual model outputs:
            'ppo': float in [-1,1] (or probability margin mapped to [-1,1])
            'momentum': float in [-1,1]
            'meanrev': float in [-1,1]
        Convention: positive => bullish; negative => bearish magnitude.
        """
        raise NotImplementedError("Wire this up to your live signal producers.")

    # -----------------------------
    # Blending
    # -----------------------------
    def blend_signals(self, symbol: str, signals: Dict[str, float]) -> float:
        """
        Blend PPO / Momentum / MeanRev using adaptive weights.
        Returns a scalar blended signal in [-1,1].
        """
        ppo = float(signals.get("ppo", 0.0))
        momentum = float(signals.get("momentum", 0.0))
        meanrev = float(signals.get("meanrev", 0.0))

        if self.enable_adaptive_weights:
            w = self.aw.get_weights(symbol)  # {'ppo':a, 'momentum':b, 'meanrev':c}
        else:
            w = {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3}

        self._last_weights[symbol] = w

        blended = w["ppo"] * ppo + w["momentum"] * momentum + w["meanrev"] * meanrev

        # Optional: squeeze to [-1, 1] if any overshoot (shouldn't happen if signals are bounded)
        blended = float(np.clip(blended, -1.0, 1.0))
        return blended

    # -----------------------------
    # Execution sizing (example)
    # -----------------------------
    def size_position(self, symbol: str, blended_signal: float, base_risk: float = 0.01) -> float:
        """
        Map blended signal → target position weight (e.g., -1..+1 scaled by base_risk).
        You can augment with volatility targeting elsewhere.
        """
        return float(np.clip(blended_signal, -1.0, 1.0)) * base_risk

    # -----------------------------
    # Feedback loop (critical)
    # -----------------------------
    def record_strategy_outcomes(
        self,
        symbol: str,
        realized_symbol_return: float,
        signals_used: Optional[Dict[str, float]] = None,
        autosave: bool = True,
        attribution: str = "proportional",
    ) -> Dict[str, float]:
        """
        Feed realized outcomes back into AdaptiveWeights.

        Args:
            realized_symbol_return: realized return for the bar/period (e.g., next-close/prev-close - 1)
            signals_used: the exact per-strategy signals used earlier for blending, e.g.
                          {'ppo': 0.6, 'momentum': -0.2, 'meanrev': 0.4}
                          If None, will use last known weights to apportion.
            attribution: 'proportional' | 'sign_only' | 'equal' — choose how to attribute
                         the realized return back to each strategy.
                         - proportional: attr ~ |signal| normalized by sum of |signals|
                         - sign_only: attr ~ 1 if signal aligns with realized move sign, else 0
                         - equal: splits equally regardless of signals
        Returns:
            New weights dict after update.
        """

        if not self.enable_adaptive_weights:
            return {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3}

        # Strategy keys we support
        keys = ["ppo", "momentum", "meanrev"]

        # If signals not provided, derive a simple attribution from last weights
        if signals_used is None:
            w = self._last_weights.get(symbol, {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3})
            sigs = {k: float(w.get(k, 0.0)) for k in keys}
        else:
            sigs = {k: float(signals_used.get(k, 0.0)) for k in keys}

        # Build per-strategy returns to feed the engine
        if attribution == "proportional":
            mags = {k: abs(v) for k, v in sigs.items()}
            s = sum(mags.values())
            if s <= 0:
                shares = {k: 1/3 for k in keys}
            else:
                shares = {k: v / s for k, v in mags.items()}
        elif attribution == "sign_only":
            # Credit only those aligned with realized move sign
            move_sign = 1.0 if realized_symbol_return >= 0 else -1.0
            aligned = {k: 1.0 if math.copysign(1.0, sigs[k] if sigs[k] != 0 else move_sign) == move_sign else 0.0 for k in keys}  # type: ignore
            s = sum(aligned.values())
            shares = {k: (aligned[k] / s if s > 0 else 1/3) for k in keys}
        else:  # 'equal'
            shares = {k: 1/3 for k in keys}

        per_strategy_returns = {k: shares[k] * realized_symbol_return for k in keys}

        # Update the adaptive engine
        new_w = self.aw.update(symbol, returns=per_strategy_returns, autosave=autosave)
        self._last_weights[symbol] = new_w
        return new_w

    # -----------------------------
    # Convenience: debug view
    # -----------------------------
    def get_last_weights(self, symbol: str) -> Dict[str, float]:
        return dict(self._last_weights.get(symbol, {"ppo": 1/3, "momentum": 1/3, "meanrev": 1/3}))
    
        # --------------------------------------------------------
    # Dashboard summary export
    # --------------------------------------------------------
    def export_summary(self) -> pd.DataFrame:
        """
        Returns a dashboard-friendly summary DataFrame with
        latest weights and key feature scores.
        """
        rows = []
        for s in self.symbols:
            feats = self._compute_features(s)
            rows.append({
                "Symbol": s,
                "Momentum": feats["momentum"],
                "MeanRev": feats["meanrev"],
                "Volatility": feats["volatility"],
                "Weight": self.last_allocs.get(s, 0.0)
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
        return df
