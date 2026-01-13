from __future__ import annotations
from typing import Dict, Any

def sanitize_allocation(symbol: str, qty: float, price: float, equity: float,
                        round_lots: int = 1,
                        min_notional: float = 10.0,
                        max_notional_pct_equity: float = 15.0) -> Dict[str, Any]:
    qty = max(0.0, float(qty))
    # Enforce max notional
    max_notional = (max_notional_pct_equity / 100.0) * max(equity, 1e-9)
    if qty * price > max_notional and price > 0:
        qty = max_notional / price

    # Enforce min notional (rounding might zero out small orders)
    if qty * price < min_notional:
        return {"qty": 0.0, "dropped": True, "reason": "min_notional"}

    # Round lots
    if round_lots > 1:
        qty = int(qty // round_lots) * round_lots
    else:
        qty = int(qty)  # default integer shares

    return {"qty": float(qty), "dropped": qty == 0}
