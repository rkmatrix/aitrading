# -*- coding: utf-8 -*-
"""
Phase 19 → Phase 20 Integration Bridge
Connects the adaptive execution controller (Phase 19)
with the RL Execution Agent (Phase 20).
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from ai.execution.executor_env import EnvConfig

@dataclass
class AdaptiveInputs:
    volatility: float
    liquidity: float
    risk_score: float
    spread_bps: float

@dataclass
class AdaptiveParams:
    slip_scale: float
    fee_bps: float
    action_buckets: List[Tuple[float, float]]

class AdaptiveExecutionBridge:
    """Map adaptive controller signals → RL environment configuration."""

    @staticmethod
    def derive_params(inputs: AdaptiveInputs) -> AdaptiveParams:
        # Slippage scaling increases with volatility & spread
        slip_scale = 0.3 + 0.7 * min(1.0, (inputs.volatility + inputs.spread_bps / 10) / 5)
        # Fees increase slightly if liquidity is poor
        fee_bps = 0.05 + 0.05 * (1 - min(inputs.liquidity / 100, 1.0))
        # Adjust aggressiveness: riskier market → smaller slices, lower aggressiveness
        risk_factor = max(0.2, 1 - inputs.risk_score)
        base_buckets = [
            (0.00, 0.00),
            (0.05 * risk_factor, 0.25 * risk_factor),
            (0.10 * risk_factor, 0.50 * risk_factor),
            (0.20 * risk_factor, 0.75 * risk_factor),
            (0.35 * risk_factor, 1.00 * risk_factor),
        ]
        return AdaptiveParams(slip_scale=slip_scale, fee_bps=fee_bps, action_buckets=base_buckets)

    @staticmethod
    def update_env_config(env_cfg: EnvConfig, adaptive_params: AdaptiveParams) -> EnvConfig:
        env_cfg.slip_scale = adaptive_params.slip_scale
        env_cfg.fee_bps = adaptive_params.fee_bps
        env_cfg.action_buckets = adaptive_params.action_buckets
        return env_cfg

    @staticmethod
    def apply_phase19_signals(signals: Dict[str, Any], env_cfg: EnvConfig) -> EnvConfig:
        """Use dict-like signals from Phase 19 output."""
        inputs = AdaptiveInputs(
            volatility=signals.get("volatility", 1.0),
            liquidity=signals.get("liquidity", 100.0),
            risk_score=signals.get("risk_score", 0.5),
            spread_bps=signals.get("spread_bps", 1.0),
        )
        params = AdaptiveExecutionBridge.derive_params(inputs)
        return AdaptiveExecutionBridge.update_env_config(env_cfg, params)
