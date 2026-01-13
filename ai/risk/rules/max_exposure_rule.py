from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class MaxExposureRule(RiskRule):
    name = "max_exposure"

    def __init__(self, long_gross_limit_pct: float, short_gross_limit_pct: float, net_limit_pct: float):
        self.long_limit = long_gross_limit_pct / 100.0
        self.short_limit = short_gross_limit_pct / 100.0
        self.net_limit = net_limit_pct / 100.0

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        # Expect ctx to include "gross_long", "gross_short", "net_exposure", "intent_notional"
        gross_long = ctx.get("gross_long", 0.0)
        gross_short = ctx.get("gross_short", 0.0)
        net = ctx.get("net_exposure", 0.0)
        eq = max(snapshot.equity, 1e-9)
        intent_notional = ctx.get("intent_notional", 0.0)

        future_long = gross_long + (intent_notional if intent.side in ("buy", "cover") else 0.0)
        future_short = gross_short + (intent_notional if intent.side in ("short", "sell") else 0.0)
        future_net = net + (intent_notional if intent.side in ("buy", "cover") else -intent_notional)

        if future_long / eq > self.long_limit:
            return RuleDecision(False, "Long gross exposure limit", {"future_long_pct": future_long / eq})
        if future_short / eq > self.short_limit:
            return RuleDecision(False, "Short gross exposure limit", {"future_short_pct": future_short / eq})
        if abs(future_net) / eq > self.net_limit:
            return RuleDecision(False, "Net exposure limit", {"future_net_pct": abs(future_net) / eq})

        return RuleDecision(True, "OK", {"future_long": future_long, "future_short": future_short, "future_net": future_net})
