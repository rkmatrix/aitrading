from __future__ import annotations
import math, time
from typing import List, Dict
from ai.execution.exec_styles.base import ExecStyle, MarketState
from ai.execution.exec_styles.scheduler import SimpleTicker
from ai.execution.order_types import OrderRequest

def _target_fraction(elapsed: float, horizon: float) -> float:
    """Crude intrahorizon volume curve (front-loaded mild S-curve)."""
    x = max(0.0, min(1.0, elapsed / max(1.0, horizon)))
    return 0.5*(1 - math.cos(math.pi * x))  # 0→1 smooth

class VWAPStyle(ExecStyle):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.horizon = float(self.params.get("horizon_sec", 900))
        self.tick = SimpleTicker(self.params.get("refresh_sec", 5))
        self.start_ts = time.time()

    def next_children(self, state: MarketState) -> List[OrderRequest]:
        if not self.tick.due() or self.done(state):
            return []
        elapsed = time.time() - self.start_ts
        target_frac = _target_fraction(elapsed, self.horizon)
        # desired cumulative vs sent so far inferred by remaining
        max_child = int(self.params.get("max_child_qty", 25))
        send_qty = max(1, min(state.remaining_qty, max_child))
        # passive limit near best
        off_bps = float(self.params.get("price_offset_bps", 3))
        q = state.quote
        if state.side == "BUY":
            limit_px = min(q["ask"], q["bid"] * (1 + off_bps/10000.0))
        else:
            limit_px = max(q["bid"], q["ask"] * (1 - off_bps/10000.0))

        return [OrderRequest(
            symbol=state.symbol,
            side=state.side,
            qty=send_qty,
            order_type="LIMIT",
            limit_price=round(limit_px, 2),
            tif="DAY",
            meta={"exec":"VWAP","elapsed":elapsed,"target_frac":target_frac}
        )]

    def done(self, state: MarketState) -> bool:
        return super().done(state) or (time.time() - self.start_ts) >= self.horizon
