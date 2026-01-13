"""
ai/policy/agents/regime_risk_agent.py
-------------------------------------
Regime / risk overlay agent.

Uses volatility, drawdown, and basic signals to scale exposure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RegimeRiskAgent:
    def __init__(
        self,
        *,
        symbols,
        high_vol_threshold: float = 0.6,
        deep_dd_threshold: float = 0.15,
        name: str = "regime_risk",
    ) -> None:
        self.symbols = list(symbols)
        self.high_vol_threshold = float(high_vol_threshold)
        self.deep_dd_threshold = float(deep_dd_threshold)
        self.name = name

    def decide(
        self,
        obs: Dict[str, Any],
        *,
        portfolio: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if info is None:
            info = {}

        vol = (
            info.get("realized_volatility")
            or obs.get("realized_volatility")
            or 0.2
        )
        drawdown = (
            info.get("drawdown")
            or obs.get("drawdown")
            or (portfolio or {}).get("drawdown")
            or 0.0
        )

        # Base exposure multiplier (1 = neutral)
        exposure = 1.0

        if vol > self.high_vol_threshold:
            exposure *= 0.5
        if drawdown > self.deep_dd_threshold:
            exposure *= 0.5

        # direction-neutral, just scales exposure
        action = {sym: float(exposure) for sym in self.symbols}

        risk_score = 0.2 if exposure < 1.0 else 0.4

        return {
            "action": action,
            "confidence": 0.8,
            "risk_score": risk_score,
            "meta": {
                "volatility": float(vol),
                "drawdown": float(drawdown),
                "exposure": float(exposure),
            },
        }
