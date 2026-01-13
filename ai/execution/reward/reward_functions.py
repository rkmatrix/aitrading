def implementation_shortfall(arrival_px: float, fills: list[tuple[int, float]], side: int) -> float:
    """fills: list of (qty, px). side +1 buy, -1 sell"""
    import numpy as np
    if not fills: return 0.0
    qty = sum(q for q,_ in fills)
    if qty == 0: return 0.0
    vwap = sum(q*p for q,p in fills)/qty
    return (vwap - arrival_px) * qty * side
