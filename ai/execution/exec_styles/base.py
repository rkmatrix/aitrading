from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from ai.execution.order_types import OrderRequest

@dataclass
class MarketState:
    symbol: str
    side: str
    remaining_qty: int
    quote: Dict[str, float]  # {bid, ask, last, size}
    now_ts: float

class ExecStyle:
    """Abstract execution style → emits child orders over time."""
    def __init__(self, params: Dict):
        self.params = params or {}

    def next_children(self, state: MarketState) -> List[OrderRequest]:
        raise NotImplementedError

    def done(self, state: MarketState) -> bool:
        return state.remaining_qty <= 0
