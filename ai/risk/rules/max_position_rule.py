from __future__ import annotations
from typing import Dict, Any
from .base import RiskRule, RuleDecision, OrderIntent, PortfolioSnapshot

class MaxPositionRule(RiskRule):
    name = "max_position"

    def __init__(self, max_pct_equity_per_symbol: float = 0.15, max_shares_per_symbol: float | None = None):
        self.max_pct = max_pct_equity_per_symbol
        self.max_shares = max_shares_per_symbol

    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision:
        eq = max(snapshot.equity, 1e-9)
        last_price = ctx.get("last_price", {}).get(intent.symbol)
        if last_price is None:
            return RuleDecision(False, f"Missing last price for {intent.symbol}")

        existing_qty = snapshot.positions.get(intent.symbol, {}).get("qty", 0)
        future_qty = existing_qty + (intent.qty if intent.side in ("buy", "cover") else -intent.qty)

        notional = abs(future_qty) * float(last_price)
        limit = self.max_pct * eq

        if self.max_shares is not None and abs(future_qty) > self.max_shares:
            return RuleDecision(False, f"Max shares exceeded ({future_qty} > {self.max_shares})",
                                {"future_qty": future_qty, "max_shares": self.max_shares})

        if notional > limit:
            return RuleDecision(False, f"Max notional exceeded ({notional:.2f} > {limit:.2f})",
                                {"future_notional": notional, "limit": limit})

        return RuleDecision(True, "OK", {"future_notional": notional, "limit": limit})
