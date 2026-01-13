from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class VolatilityThrottleRule(RiskRule):
    name = "volatility_throttle"

    def __init__(self, vix_threshold: float, reduce_size_factor: float):
        self.vix_threshold = vix_threshold
        self.reduce_size_factor = reduce_size_factor

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        # Expect ctx["vix"] = float and ctx["mutations"]["qty_factor"] to be mutable
        vix = ctx.get("vix")
        if vix is None:
            return RuleDecision(True, "No VIX provided; skipping")

        if vix >= self.vix_threshold:
            # Suggest a mutation on qty via context (guardian will apply)
            mutations = ctx.setdefault("mutations", {})
            mutations["qty_factor"] = min(mutations.get("qty_factor", 1.0), self.reduce_size_factor)
            return RuleDecision(True, f"High VIX {vix:.2f} >= {self.vix_threshold:.2f}; throttling size",
                                {"qty_factor": mutations["qty_factor"]})
        return RuleDecision(True, "OK", {"vix": vix})
