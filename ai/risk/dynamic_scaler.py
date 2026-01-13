"""
Phase 92.2 - Dynamic Exposure Scaling (DES)
AITradeBot

This module computes a dynamic scaling factor applied inside the
RiskEnvelopeController before evaluating exposure caps.

Returned value is between 0.0 and 1.5:
    0.0  → force kill
    0.25 → minimum exposure
    0.50 → half exposure
    1.00 → normal exposure
    1.50 → aggressive boost (rare)
"""

from __future__ import annotations
import math


class DynamicExposureScaler:

    def compute_scale(self, ctx: dict) -> float:
        """
        ctx contains:
            equity
            volatility
            drawdown
            trend_strength
            ml_variance
            concentration
            leverage
            regime
        """

        vol = float(ctx.get("volatility", 1.0))
        dd = float(ctx.get("drawdown", 0.0))
        trend = float(ctx.get("trend_strength", 0.0))
        var = float(ctx.get("ml_variance", 0.0))
        lev = float(ctx.get("leverage", 1.0))
        conc = float(ctx.get("concentration", 0.0))
        regime = ctx.get("regime", "unknown")

        # -----------------------------------------
        # BASELINE (regime map)
        # -----------------------------------------
        baseline = {
            "quiet_trend": 1.00,
            "rangebound": 0.85,
            "volatile_trend": 0.70,
            "chaos": 0.50,
            "extreme_vol": 0.00,
            "unknown": 1.00,
        }.get(regime, 1.00)

        scale = baseline

        # -----------------------------------------
        # Trend signal → increases scale
        # -----------------------------------------
        if trend > 0:
            scale += min(0.25, trend * 0.30)

        # -----------------------------------------
        # Volatility increases → reduce scale
        # -----------------------------------------
        if vol > 1.5:
            scale -= min(0.40, (vol - 1.5) * 0.25)

        # -----------------------------------------
        # Drawdown → reduce
        # -----------------------------------------
        if dd > 0.04:
            scale -= min(0.50, dd * 4.0)

        # -----------------------------------------
        # ML variance → uncertainty
        # -----------------------------------------
        if var > 0.002:
            scale -= min(0.30, (var - 0.002) * 100)

        # -----------------------------------------
        # Leverage pressure
        # -----------------------------------------
        if lev > 1.3:
            scale -= min(0.30, (lev - 1.3) * 0.50)

        # -----------------------------------------
        # Concentration per symbol
        # -----------------------------------------
        if conc > 0.60:
            scale -= min(0.25, (conc - 0.60) * 0.40)

        # Clamp limits
        scale = max(0.0, min(1.5, scale))

        return scale
