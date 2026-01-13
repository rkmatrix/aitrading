import logging
from collections import defaultdict
from typing import Dict, Any, Optional, Iterable

logger = logging.getLogger("OrderStateTracker")

class OrderStateTracker:
    """
    Tracks live orders and session PnL.
    Integrates with broker/router get_status() or get_order().
    """
    def __init__(self):
        self.orders: Dict[str, Dict[str, Any]] = {}
        self._pnl: float = 0.0

    def track(self, order: Dict[str, Any]) -> str:
        oid = str(order.get("id") or order.get("order_id") or f"tmp_{len(self.orders)}")
        self.orders[oid] = order
        logger.info(f"🧾 Tracking order {oid}: {order}")
        return oid

    def iter_open_orders(self) -> Iterable[str]:
        for oid, od in self.orders.items():
            status = str(od.get("status", "new")).lower()
            if status in ("new", "accepted", "pending_new", "open", "partially_filled"):
                yield oid

    def refresh_with_router(self, router: Any) -> None:
        getter = None
        for cand in ("get_order", "get_status", "status"):
            if hasattr(router, cand):
                getter = getattr(router, cand)
                break
        if getter is None:
            logger.debug("Router has no status getter; skipping refresh.")
            return
        for oid in list(self.orders.keys()):
            try:
                st = getter(oid)
                if isinstance(st, dict):
                    self.orders[oid].update(st)
            except Exception as e:
                logger.debug(f"Status refresh failed for {oid}: {e}")

    def current_pnl(self) -> float:
        return self._pnl

    def set_pnl(self, value: float) -> None:
        self._pnl = float(value)

    def get(self, oid: str) -> Optional[Dict[str, Any]]:
        return self.orders.get(oid)
