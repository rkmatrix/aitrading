from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class EarningsHaltRule(RiskRule):
    name = "earnings_halt"

    def __init__(self, pre_earnings_halt_days: int = 2, post_earnings_halt_days: int = 1):
        self.pre = pre_earnings_halt_days
        self.post = post_earnings_halt_days

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        # Expect ctx["earnings_days_to"] = {"AAPL": -1} (negative => days after, positive => days before)
        dmap = ctx.get("earnings_days_to", {})
        d = dmap.get(intent.symbol)
        if d is None:
            return RuleDecision(True, "No earnings data")

        if d >= 0 and d <= self.pre:
            return RuleDecision(False, f"Earnings in {d} days (pre-halt {self.pre}d)")
        if d < 0 and abs(d) <= self.post:
            return RuleDecision(False, f"Earnings {abs(d)} days ago (post-halt {self.post}d)")

        return RuleDecision(True, "OK", {"earnings_days_to": d})
