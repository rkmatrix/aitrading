from __future__ import annotations
from typing import Dict, Any

def sanitize_order(intent: Dict[str, Any], price_slippage_limit_bps: int = 50,
                   allow_market: bool = False, time_in_force: str = "day") -> Dict[str, Any]:
    order_type = intent.get("order_type", "limit")
    side = intent.get("side")
    qty = float(intent.get("qty", 0))
    ref_price = float(intent.get("ref_price", 0))
    limit_price = intent.get("limit_price")

    if qty <= 0:
        return {**intent, "valid": False, "reason": "qty<=0"}

    if order_type == "market" and not allow_market:
        order_type = "limit"

    if order_type == "limit":
        if limit_price is None and ref_price > 0:
            slip = (price_slippage_limit_bps / 10000.0) * ref_price
            limit_price = ref_price + slip if side in ("buy", "cover") else ref_price - slip

    return {
        **intent,
        "order_type": order_type,
        "limit_price": limit_price,
        "time_in_force": time_in_force,
        "valid": True
    }
