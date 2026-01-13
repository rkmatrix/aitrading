from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Dict, Any

from ai.guardians.circuit_breakers import CircuitBreakers, RiskPolicy
from ai.guardians.order_burst_guard import OrderBurstGuard
from ai.state.pnl_tracker import DailyPnlProvider
from tools.locks import AtomicFlag

Side = Literal["BUY", "SELL"]

@dataclass
class GuardedRouterConfig:
    min_repeat_seconds: int = 10
    max_orders_per_minute: int = 5
    enable_daily_loss_limit: bool = True
    max_daily_loss: float = -250.0
    allow_flatten_only_below_limit: bool = True
    flags_dir: str = "data/runtime/flags"

@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: str = "MARKET"  # or LIMIT
    limit_price: Optional[float] = None
    tag: str = "default"        # strategy tag / correlation key
    is_flattening: bool = False # set True if this order reduces/flat positions

class GuardedRouter:
    """
    Wraps an underlying route_callable(order_dict) with:
      - idempotency (min_repeat_seconds)
      - burst guard (max_orders_per_minute)
      - risk circuit breaker (daily loss limit)
    """
    def __init__(
        self,
        cfg: GuardedRouterConfig,
        pnl_provider: DailyPnlProvider,
        route_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        self.cfg = cfg
        self.route_callable = route_callable

        self._flag = AtomicFlag(cfg.flags_dir)
        self._burst = OrderBurstGuard(cfg.max_orders_per_minute)
        self._cb = CircuitBreakers(
            RiskPolicy(
                enable_daily_loss_limit=cfg.enable_daily_loss_limit,
                max_daily_loss=cfg.max_daily_loss,
                allow_flatten_only_below_limit=cfg.allow_flatten_only_below_limit,
            ),
            pnl_provider=pnl_provider,
        )

    @staticmethod
    def _key(o: Order) -> str:
        # Key: side|symbol|qty|tag (limit_price excluded to be friendly with reattempts/partial fills)
        base = f"{o.side}|{o.symbol}|{o.qty}|{o.tag}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def route(self, order: Order) -> Dict[str, Any]:
        # 1) Circuit breaker
        allowed, reason = self._cb.check_new_order_allowed(order.side, order.is_flattening)
        if not allowed:
            return {"status": "BLOCKED", "reason": reason, "order": order.__dict__}

        # 2) Burst guard
        if not self._burst.allow():
            return {"status": "THROTTLED", "reason": "Burst guard (max per minute reached)", "order": order.__dict__}

        # 3) Idempotency
        key = self._key(order)
        if not self._flag.set_if_absent(key, ttl_seconds=self.cfg.min_repeat_seconds):
            return {"status": "DUPLICATE", "reason": f"Repeated order within {self.cfg.min_repeat_seconds}s", "order": order.__dict__}

        # 4) Route safely
        try:
            payload = {
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "type": order.order_type,
                "limit_price": order.limit_price,
                "tag": order.tag,
            }
            resp = self.route_callable(payload)  # delegate to Phase 23 SmartOrderRouter / broker adapter
            return {"status": "ROUTED", "router_response": resp, "order": order.__dict__}
        except Exception as e:
            return {"status": "ERROR", "error": str(e), "order": order.__dict__}
