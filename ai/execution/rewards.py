from __future__ import annotations

class ExecReward:
    def __init__(self, shortfall_weight=1.0, inventory_weight=5e-4,
                 latency_weight=1e-3, fill_bonus=0.05):
        self.w_s = shortfall_weight
        self.w_inv = inventory_weight
        self.w_lat = latency_weight
        self.fill_bonus = fill_bonus

    def __call__(self, *, side: int, mid_before: float, fill_price: float | None,
                 inventory: float, latency_ms: float, filled: bool) -> float:
        # Implementation shortfall in bps (positive is cost)
        if fill_price is not None and mid_before > 0:
            shortfall_bps = side * ((fill_price - mid_before) / mid_before) * 1e4
            # reward higher when shortfall is negative (price improvement)
            cost = - self.w_s * (-shortfall_bps)
        else:
            cost = 0.0

        inv_pen = - self.w_inv * abs(inventory)
        lat_pen = - self.w_lat * max(0.0, latency_ms) / 10.0
        fb = self.fill_bonus if filled else 0.0
        return float(cost + inv_pen + lat_pen + fb)
