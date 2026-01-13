def bps(x: float) -> float:
    return 10000.0 * x

def is_bps(arrival_px: float, fills: list[tuple[int,float]], side:int) -> float:
    if not fills: return 0.0
    qty = sum(q for q, _ in fills)
    vwap = sum(q*p for q,p in fills)/qty
    slip = (vwap - arrival_px) * side / arrival_px
    return bps(slip)
