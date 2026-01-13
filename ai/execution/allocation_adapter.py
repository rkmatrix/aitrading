# SPDX-License-Identifier: MIT
from __future__ import annotations
import json, math, logging
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


class AllocationAdapter:
    """
    Translates Phase30 target weights → concrete order instructions
    that Phase26/28 execution modules can route.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.inputs = cfg.get("inputs", {})
        self.outputs = cfg.get("outputs", {})
        self.params = cfg.get("params", {})

    # ---------------------------------------------------------------------- #
    def _load_json(self, path: str | Path, default: Any) -> Any:
        p = Path(path)
        if not p.exists():
            return default
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _get_equity(self, balances: Dict[str, Any]) -> float:
        for k in ("equity", "net_liquidation", "total_value"):
            if k in balances and isinstance(balances[k], (int, float)):
                return float(balances[k])
        # fallback to cash
        return float(balances.get("cash", 0.0))

    # ---------------------------------------------------------------------- #
    def run_once(self) -> List[Dict[str, Any]]:
        targets = self._load_json(self.inputs["targets_path"], default={}).get("targets", {})
        prices = self._load_json(self.inputs["prices_path"], default={})
        balances = self._load_json(self.inputs["equity_path"], default={})
        positions = self._load_json(self.inputs["positions_path"], default={})

        equity = self._get_equity(balances)
        min_trade_value = float(self.params.get("min_trade_value_usd", 100.0))
        max_order_notional_pct = float(self.params.get("max_order_notional_pct", 5.0))
        price_fallback = float(self.params.get("price_fallback", 1.0))
        round_lots = bool(self.params.get("round_lots", True))

        orders: List[Dict[str, Any]] = []

        for sym, target_weight in targets.items():
            price = float(prices.get(sym, price_fallback))
            pos = positions.get(sym, {})
            qty_now = float(pos.get("qty", 0.0))

            target_value = target_weight * equity
            target_qty = target_value / price
            delta = target_qty - qty_now
            side = "BUY" if delta > 0 else "SELL"

            notional = abs(delta) * price
            if notional < min_trade_value:
                continue
            if notional > (equity * max_order_notional_pct / 100.0):
                # clip large orders
                allowed_qty = (equity * max_order_notional_pct / 100.0) / price
                delta = math.copysign(allowed_qty, delta)

            qty_final = round(delta) if round_lots else delta
            if abs(qty_final) < 1e-6:
                continue

            orders.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": abs(qty_final),
                    "order_type": "MARKET",
                    "limit_price": None,
                    "tag": "phase30_adapter",
                }
            )

        out_path = Path(self.outputs["orders_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"orders": orders}, f, indent=2)

        logger.info("📦 Generated %d orders → %s", len(orders), out_path)
        return orders
