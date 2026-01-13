from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class DrawdownCircuitRule(RiskRule):
    name = "drawdown_circuit"

    def __init__(self, intraday_dd_limit_pct: float, daily_dd_limit_pct: float, cooloff_minutes: int):
        self.intraday = intraday_dd_limit_pct / 100.0
        self.daily = daily_dd_limit_pct / 100.0
        self.cooloff_minutes = cooloff_minutes

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        # Expect ctx["drawdown"] = {"intraday": 0.032, "daily": 0.041, "cooloff_active": False}
        dd = ctx.get("drawdown", {})
        intraday = dd.get("intraday", 0.0)
        daily = dd.get("daily", 0.0)
        cooloff_active = dd.get("cooloff_active", False)

        if cooloff_active:
            return RuleDecision(False, f"Cooloff active ({self.cooloff_minutes} min)")

        if intraday >= self.intraday:
            return RuleDecision(False, f"Intraday drawdown {intraday:.2%} >= {self.intraday:.2%}")
        if daily >= self.daily:
            return RuleDecision(False, f"Daily drawdown {daily:.2%} >= {self.daily:.2%}")

        return RuleDecision(True, "OK", {"intraday": intraday, "daily": daily})
