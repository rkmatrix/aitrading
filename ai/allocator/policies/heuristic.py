# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Dict, List, Any
import math

from .base import AllocationPolicy


class HeuristicPolicy(AllocationPolicy):
    """
    Simple, robust baseline allocation:
    - Convert signal strength into positive weights (rank -> weight)
    - Respect min/max weight per name
    - Enforce a cash floor by scaling down gross allocation
    """

    def __init__(
        self,
        risk_aversion: float = 0.75,  # 0..1 (higher = smaller gross)
        min_weight: float = 0.00,
        max_weight: float = 0.60,
        cash_floor: float = 0.05,
    ):
        self.risk_aversion = float(max(0.0, min(1.0, risk_aversion)))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.cash_floor = float(cash_floor)

    def _signal_to_score(self, s: Any) -> float:
        # Accept various signal shapes: number, dict with "score", etc.
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, dict):
            for key in ("score", "rank", "value", "signal"):
                if key in s and isinstance(s[key], (int, float)):
                    return float(s[key])
        return 0.0

    def propose_weights(
        self,
        *,
        symbols: List[str],
        signals: Dict[str, Any],
        balances: Dict[str, Any],
        positions: Dict[str, Any],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {sym: self._signal_to_score(signals.get(sym)) for sym in symbols}

        # Make all non-negative by shifting if needed
        min_score = min(scores.values()) if scores else 0.0
        if min_score < 0:
            scores = {k: v - min_score for k, v in scores.items()}

        total = sum(scores.values()) or 1.0
        raw = {k: v / total for k, v in scores.items()}  # base proportional weights

        # Risk aversion reduces gross exposure (convex)
        gross_target = max(0.0, 1.0 - self.risk_aversion * 0.5)  # 0.5..1.0 gross
        gross_target = max(0.0, gross_target - self.cash_floor)  # leave cash

        # Apply per-name bounds and scale to gross_target
        # First clamp to [min_weight, max_weight] as a soft preparation
        pre = {k: max(self.min_weight, min(self.max_weight, w)) for k, w in raw.items()}

        # Renormalize to 1.0 then scale to gross_target
        s = sum(pre.values()) or 1.0
        normalized = {k: v / s for k, v in pre.items()}
        final = {k: v * gross_target for k, v in normalized.items()}

        return final
