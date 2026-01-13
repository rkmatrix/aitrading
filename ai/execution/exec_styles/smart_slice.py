from __future__ import annotations
import random
from typing import List, Dict
from ai.execution.exec_styles.base import ExecStyle, MarketState
from ai.execution.exec_styles.scheduler import SimpleTicker
from ai.execution.order_types import OrderRequest

class SmartSliceStyle(ExecStyle):
    """Time-sliced with randomized qty; passive by default; post-only optional."""
    def __init__(self, params: Dict):
        super().__init__(params)
        self.tick = SimpleTicker(self.params.get("slice_sec", 5))

    def next_children(self, state: MarketState) -> List[OrderRequest]:
        if not self.tick.due() or self.done(state):
            return []
        max_child = int(self.params.get("max_child_qty", 15))
        rnd_bps = float(self.params.get("slice_random_bps", 20))
        qty = max(1, int(max_child * (1 + random.uniform(-rnd_bps, rnd_bps)/100.0)))
        qty = min(qty, state.remaining_qty)
        off_bps = float(self.params.get("price_offset_bps", 2))
        post_only = bool(self.params.get("post_only", True))

        q = state.quote
        if state.side == "BUY":
            px = q["bid"] * (1 + off_bps/10000.0) if post_only else q["ask"]
        else:
            px = q["ask"] * (1 - off_bps/10000.0) if post_only else q["bid"]

        return [OrderRequest(
            symbol=state.symbol,
            side=state.side,
            qty=qty,
            order_type="LIMIT",
            limit_price=round(px, 2),
            tif="DAY",
            meta={"exec":"SMART_SLICE","post_only":post_only}
        )]
