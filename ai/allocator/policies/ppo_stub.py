# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Dict, List, Any

from .base import AllocationPolicy


class PPOPolicyStub(AllocationPolicy):
    """
    Placeholder for a PPO-based allocator. Implement by wiring to:
    - ai.env.allocator_env.AllocatorEnv (Gymnasium-compatible)
    - stable_baselines3 PPO (or your in-house PPO)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def propose_weights(
        self,
        *,
        symbols: List[str],
        signals: Dict[str, Any],
        balances: Dict[str, Any],
        positions: Dict[str, Any],
    ) -> Dict[str, float]:
        # For now, mirror heuristic via uniform split across positive signals
        positive = {s: v for s, v in signals.items() if _score(v) > 0}
        n = len(positive) or len(symbols) or 1
        w = 1.0 / n * 0.9  # keep some cash
        return {s: (w if s in positive else 0.0) for s in symbols}


def _score(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        for key in ("score", "rank", "value", "signal"):
            if key in val and isinstance(val[key], (int, float)):
                return float(val[key])
    return 0.0
