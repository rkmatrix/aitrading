from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class NewsHaltRule(RiskRule):
    name = "news_halt"

    def __init__(self, severity_threshold: str = "high"):
        self.severity_threshold = severity_threshold

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        # Expect ctx["news_alerts"] = [{"symbol":"AAPL","severity":"high","headline":"..."}]
        alerts = ctx.get("news_alerts", [])
        sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        need = sev_rank.get(self.severity_threshold, 2)

        for a in alerts:
            if a.get("symbol") == intent.symbol and sev_rank.get(a.get("severity", "low"), 0) >= need:
                return RuleDecision(False, f"News halt: {a.get('severity')}", {"headline": a.get("headline")})

        return RuleDecision(True, "OK")
