"""
ai/reward/execution_aware_reward.py
-----------------------------------

Phase 75 — Execution-Aware Reward Engine

This combines:
    - Market PnL reward
    - Execution slippage costs
    - Spread costs
    - Latency penalty
    - Fill-probabilistic reward
    - Prediction warnings/clamps/blocks
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExecutionAwareReward:
    """
    Core reward model for Phase 75.
    """

    def __init__(
        self,
        *,
        slippage_weight: float = -1.0,      # negative weight (penalty)
        spread_weight: float = -0.5,        # penalty
        latency_weight: float = -0.0005,    # penalty per ms
        fill_prob_weight: float = 1.0,      # reward for trades with higher fill-prob
        clamp_penalty: float = -0.3,        # penalty if router clamps qty
        pred_block_penalty: float = -2.0,   # penalty if prediction blocks
        risk_block_penalty: float = -3.0,   # penalty if risk envelope blocks
    ) -> None:
        self.slippage_weight = slippage_weight
        self.spread_weight = spread_weight
        self.latency_weight = latency_weight
        self.fill_prob_weight = fill_prob_weight
        self.clamp_penalty = clamp_penalty
        self.pred_block_penalty = pred_block_penalty
        self.risk_block_penalty = risk_block_penalty

    # ------------------------------------------------------------------
    # Main combine function
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        *,
        pnl: float,
        exec_info: Dict[str, Any],
        pred_info: Optional[Dict[str, Any]] = None,
        router_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        pnl: realized or next-step reward from environment
        exec_info: slippage, spread, latency
        pred_info: prediction model output
        router_info: clamp/block info
        """

        reward = pnl

        # --------------------- Slippage + Spread -----------------------
        sl = exec_info.get("slippage")
        if sl is not None:
            reward += self.slippage_weight * sl

        sp = exec_info.get("spread")
        if sp is not None:
            reward += self.spread_weight * sp

        # --------------------- Latency penalty -------------------------
        lat = exec_info.get("latency_ms")
        if lat is not None:
            reward += self.latency_weight * lat

        # --------------------- Prediction signals ----------------------
        if pred_info:
            fill_prob = pred_info.get("pred_fill_prob")
            if fill_prob is not None:
                reward += self.fill_prob_weight * fill_prob

        # --------------------- Router clamp/block ----------------------
        if router_info:
            if router_info.get("clamped_by_prediction"):
                reward += self.clamp_penalty

            if router_info.get("prediction_blocked"):
                reward += self.pred_block_penalty

            if router_info.get("risk_blocked"):
                reward += self.risk_block_penalty

        return reward
