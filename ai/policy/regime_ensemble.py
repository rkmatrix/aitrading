# ai/policy/regime_ensemble.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import copysign
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Regime Weight Definitions
# ============================================================
@dataclass
class RegimeWeights:
    rl_weight: float
    trend_weight: float
    revert_weight: float
    stay_flat_if_disagree: bool = False
    stay_flat_threshold: float = 0.3


# ============================================================
# Ensemble Configuration (with default_factory FIXED)
# ============================================================
@dataclass
class RegimeEnsembleConfig:
    enabled: bool = True
    use_regime: bool = True
    clamp_small_scores: float = 0.15

    risk_on: RegimeWeights = field(
        default_factory=lambda: RegimeWeights(
            rl_weight=0.7,
            trend_weight=0.2,
            revert_weight=0.1,
            stay_flat_if_disagree=False,
            stay_flat_threshold=0.3,
        )
    )

    neutral: RegimeWeights = field(
        default_factory=lambda: RegimeWeights(
            rl_weight=0.5,
            trend_weight=0.25,
            revert_weight=0.25,
            stay_flat_if_disagree=False,
            stay_flat_threshold=0.3,
        )
    )

    risk_off: RegimeWeights = field(
        default_factory=lambda: RegimeWeights(
            rl_weight=0.2,
            trend_weight=0.3,
            revert_weight=0.5,
            stay_flat_if_disagree=True,
            stay_flat_threshold=0.3,
        )
    )


# ============================================================
# Regime-Aware Ensemble Engine
# ============================================================
class RegimeEnsemble:
    """
    Phase 121 – Regime-aware ensemble of:
        - RL/fusion score
        - Dumb trend-follow score
        - Dumb mean-revert score
    """

    def __init__(self, cfg: RegimeEnsembleConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _weights_for_regime(self, regime_name: str) -> RegimeWeights:
        if regime_name.upper() == "RISK_ON":
            return self.cfg.risk_on
        if regime_name.upper() == "RISK_OFF":
            return self.cfg.risk_off
        return self.cfg.neutral

    # ------------------------------------------------------------------
    def combine(
        self,
        *,
        regime_state: Optional[Any],
        rl_score: float,
        dumb_scores: Dict[str, float],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "ensemble_score": float,
                "mode": "...",
                "weights": {...},
                "components": {...},
                "regime_name": "...",
            }
        """

        # If disabled → pass through RL score
        if not self.cfg.enabled:
            return {
                "ensemble_score": rl_score,
                "mode": "RL_ONLY",
                "weights": {"rl": 1.0, "trend": 0.0, "revert": 0.0},
                "components": {
                    "rl": rl_score,
                    "trend": dumb_scores.get("trend_score", 0.0),
                    "revert": dumb_scores.get("revert_score", 0.0),
                },
                "regime_name": getattr(regime_state, "name", "UNKNOWN")
                if regime_state else "UNKNOWN",
            }

        regime_name = (
            regime_state.name
            if (self.cfg.use_regime and regime_state)
            else "NEUTRAL"
        )

        weights = self._weights_for_regime(regime_name)

        rl = float(rl_score)
        trend = float(dumb_scores.get("trend_score", 0.0))
        revert = float(dumb_scores.get("revert_score", 0.0))

        components = {"rl": rl, "trend": trend, "revert": revert}

        # ===========================
        # Linear blend
        # ===========================
        raw = (
            weights.rl_weight * rl
            + weights.trend_weight * trend
            + weights.revert_weight * revert
        )

        mode = "BLEND"

        # ===========================
        # Risk-Off “disagreement stay-flat” logic
        # ===========================
        if regime_name.upper() == "RISK_OFF" and weights.stay_flat_if_disagree:
            dumb_combo = trend + revert

            if (
                abs(dumb_combo) >= weights.stay_flat_threshold
                and abs(rl) >= weights.stay_flat_threshold
                and copysign(1.0, dumb_combo) != copysign(1.0, rl)
            ):
                # Strong disagreement → stay flat
                logger.warning(
                    "RegimeEnsemble: %s RISK_OFF disagreement RL=%.3f dumb=%.3f → STAY_FLAT",
                    symbol,
                    rl,
                    dumb_combo,
                )
                return {
                    "ensemble_score": 0.0,
                    "mode": "STAY_FLAT",
                    "weights": {"rl": 0, "trend": 0, "revert": 0},
                    "components": components,
                    "regime_name": regime_name,
                }

        # ===========================
        # Clamp small scores
        # ===========================
        ensemble_score = float(raw)
        if abs(ensemble_score) < self.cfg.clamp_small_scores:
            ensemble_score = 0.0
            mode = "STAY_FLAT"

        # Detect RL-only situations for logging clarity
        if abs(rl) > 0.5 and abs(trend) < 0.1 and abs(revert) < 0.1:
            mode = "RL_ONLY"

        return {
            "ensemble_score": ensemble_score,
            "mode": mode,
            "weights": {
                "rl": weights.rl_weight,
                "trend": weights.trend_weight,
                "revert": weights.revert_weight,
            },
            "components": components,
            "regime_name": regime_name,
        }
