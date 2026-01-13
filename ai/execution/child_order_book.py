from dataclasses import dataclass, field
from typing import List
from ai.execution.order_types import OrderResult

@dataclass
class ChildFill:
    order_id: str
    qty: int
    price: float

@dataclass
class ChildOrderBook:
    fills: List[ChildFill] = field(default_factory=list)

    def add_fill(self, res: OrderResult):
        if res and res.ok and res.filled_qty > 0 and res.avg_fill_price is not None:
            self.fills.append(ChildFill(res.order_id, res.filled_qty, res.avg_fill_price))

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)

    @property
    def vwap(self) -> float | None:
        tot_qty = self.filled_qty
        if tot_qty == 0:
            return None
        return sum(f.qty * f.price for f in self.fills) / tot_qty
