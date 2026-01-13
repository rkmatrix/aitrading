from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

@dataclass
class RuleDecision:
    allow: bool
    reason: str = ""
    details: Optional[Dict[str, Any]] = None

class PortfolioSnapshot(Protocol):
    equity: float
    cash: float
    positions: Dict[str, Dict[str, Any]]  # {"AAPL": {"qty": 10, "avg_price": 190.5, "side": "long"}}

class OrderIntent(Protocol):
    symbol: str
    side: str           # "buy" | "sell" | "short" | "cover"
    qty: float
    order_type: str     # "limit" | "market"
    limit_price: Optional[float]

class RiskRule(Protocol):
    name: str
    def check(self, intent: OrderIntent, snapshot: PortfolioSnapshot, ctx: Dict[str, Any]) -> RuleDecision: ...
