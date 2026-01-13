# ai/execution/blend_to_orders.py
from __future__ import annotations
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class BlendOrderPlanner:
    """
    Turn blended targets in [-1..1] into concrete orders by comparing current position to desired.
    You can wire a real position/equity fetcher later; defaults are safe no-ops.
    """

    def __init__(
        self,
        symbol_prices_fn,        # fn(symbol)->last_price (float)
        get_position_fn=None,    # fn(symbol)->current_shares (int/float)
        get_equity_fn=None,      # fn()->float equity
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg or {}
        self._price = symbol_prices_fn
        self._pos = get_position_fn or (lambda sym: 0.0)
        self._equity = get_equity_fn or (lambda: float(self.cfg.get("account_equity", 100000.0)))
        self._last_order_ts: Dict[str, float] = {}

    def _cooldown_ok(self, sym: str, now: float) -> bool:
        cd = float(self.cfg.get("cooldown_s", 15.0))
        last = self._last_order_ts.get(sym, 0.0)
        return (now - last) >= cd

    def plan_orders(self, blended_targets: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Returns per-symbol order dict if an order is needed:
        {
          "AAPL": {"symbol": "AAPL", "side": "BUY", "qty": 10, "order_type": "MARKET", "tag": "..."},
          ...
        }
        """
        if self.cfg.get("kill_switch", False):
            logger.warning("🛑 Kill-switch active. Skipping all orders.")
            return {}

        now = time.time()
        orders: Dict[str, Dict[str, Any]] = {}

        max_abs = float(self.cfg.get("max_abs_target", 0.9))
        max_trade_qty = float(self.cfg.get("max_trade_qty", 0))
        min_trade_notional = float(self.cfg.get("min_trade_notional", 0.0))
        order_type = self.cfg.get("order_type", "MARKET")
        tif = self.cfg.get("time_in_force", "day")
        tag = self.cfg.get("tag", "phase53_blend")

        equity = self._equity()
        per_sym_anchor = float(self.cfg.get("target_notional_per_symbol", 0.0))

        for sym, tgt in blended_targets.items():
            tgt = float(max(min(tgt, max_abs), -max_abs))

            price = float(self._price(sym) or 0.0)
            if price <= 0:
                logger.warning("No/invalid price for %s; skipping.", sym)
                continue

            # Desired shares: either from per-symbol anchor or equity * target
            if per_sym_anchor > 0:
                desired_notional = per_sym_anchor * tgt
            else:
                desired_notional = equity * tgt / max(1, len(blended_targets))
            desired_shares = desired_notional / price

            current_shares = float(self._pos(sym))
            delta = desired_shares - current_shares

            # Round to whole shares (stock)
            qty = int(round(abs(delta)))
            if qty == 0:
                continue

            # Guardian checks
            if not self._cooldown_ok(sym, now):
                continue
            if max_trade_qty > 0 and qty > max_trade_qty:
                qty = int(max_trade_qty)
            if qty * price < min_trade_notional:
                continue

            side = "BUY" if delta > 0 else "SELL"
            orders[sym] = {
                "symbol": sym,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "time_in_force": tif,
                "tag": tag,
                "is_flattening": False,
            }
            self._last_order_ts[sym] = now

        return orders
