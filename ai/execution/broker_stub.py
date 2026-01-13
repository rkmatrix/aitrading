from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class OrderFill:
    symbol: str
    target_weight: float
    filled_weight: float
    price: float
    commission: float
    slippage: float
    ts: float


class BrokerStub:
    """
    Simulated broker connector with simple fill + cost model.
    Replace with concrete Alpaca/IB adapters later.
    """

    def __init__(self, assets: List[str], commission_per_share: float = 0.0, spread_bps: float = 2.0):
        self.assets = assets
        self.commission_per_share = commission_per_share
        self.spread_bps = spread_bps
        self.prices = {a: 100.0 for a in assets}  # mock prices

    def quote(self, symbol: str) -> float:
        # Random walk-ish price
        p = self.prices.get(symbol, 100.0) * (1.0 + np.random.normal(0, 0.0005))
        self.prices[symbol] = max(1e-3, p)
        return self.prices[symbol]

    def rebalance(self, targets: Dict[str, float]) -> Dict[str, OrderFill]:
        fills: Dict[str, OrderFill] = {}
        for sym, tw in targets.items():
            px = self.quote(sym)
            # Simple slippage: half-spread * |weight|
            slippage = abs(tw) * (self.spread_bps * 1e-4) * px
            commission = abs(tw) * self.commission_per_share  # for stubbing only
            # Partial fills: 98%-100%
            filled = tw * float(np.random.uniform(0.98, 1.0))
            fills[sym] = OrderFill(
                symbol=sym,
                target_weight=tw,
                filled_weight=filled,
                price=px,
                commission=commission,
                slippage=slippage,
                ts=time.time(),
            )
        return fills
