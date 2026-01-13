from __future__ import annotations
import time
from typing import List, Dict
from ai.execution.exec_styles.base import ExecStyle, MarketState
from ai.execution.exec_styles.scheduler import SimpleTicker
from ai.execution.order_types import OrderRequest

class POVStyle(ExecStyle):
    """Participate at X% of estimated tape flow (very simple size proxy)."""
    def __init__(self, params: Dict, baseline_vol_per_sec: float = 2500):
        super().__init__(params)
        self.participation = float(self.params.get("participation", 0.1))
        self.baseline = float(baseline_vol_per_sec)
        self.tick = SimpleTicker(self.params.get("refresh_sec", 3))

    def _estimate_flow(self, quote: Dict[str,float]) -> float:
        # crude proxy using NBBO size & spread
        size = float(quote.get("size", 100))
        spread = max(0.01, quote["ask"] - quote["bid"])
        return self.baseline * min(1.5, size/100.0) * (1.0/spread)

    def next_children(self, state: MarketState) -> List[OrderRequest]:
        if not self.tick.due() or self.done(state):
            return []
        est_flow = self._estimate_flow(state.quote)   # shares per sec
        slice_qty = max(1, int(self.participation * est_flow * self.tick.period))
        slice_qty = min(slice_qty, int(self.params.get("max_child_qty", 20)), state.remaining_qty)

        off_bps = float(self.params.get("price_offset_bps", 2))
        q = state.quote
        if state.side == "BUY":
            limit_px = min(q["ask"], q["bid"] * (1 + off_bps/10000.0))
        else:
            limit_px = max(q["bid"], q["ask"] * (1 - off_bps/10000.0))

        return [OrderRequest(
            symbol=state.symbol,
            side=state.side,
            qty=slice_qty,
            order_type="LIMIT",
            limit_price=round(limit_px, 2),
            tif="IOC",  # be a bit more aggressive than DAY
            meta={"exec":"POV","est_flow":est_flow}
        )]
