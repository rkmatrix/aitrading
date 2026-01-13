from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ai.execution.broker_stub import BrokerStub
from ai.router.alpha_router import AlphaRouter


@dataclass
class ExecResult:
    pnl: float
    costs: float
    turnover_l1: float
    drawdown: float
    kill: bool


class ExecutionManager:
    """
    Takes target weights, sends to broker, returns realized PnL/costs/turnover.
    """

    def __init__(self, assets: List[str], router: AlphaRouter, dd_kill: float = 0.10):
        self.assets = assets
        self.router = router
        self.broker = BrokerStub(assets)
        self.prev_weights = {a: 0.0 for a in assets}
        self.equity = 1_000_000.0
        self.highwater = self.equity
        self.dd_kill = dd_kill

    def step(self, allocator_weights: Dict[str, float], signals: Dict[str, float]) -> ExecResult:
        # Router has already blended direction; we receive signed targets from caller
        targets = allocator_weights  # already signed + capped at router level in runner

        # Send to broker and compute PnL / costs
        fills = self.broker.rebalance(targets)

        # Compute turnover and costs
        cur_w = {a: float(fills[a].filled_weight) for a in self.assets}
        turnover = sum(abs(cur_w[a] - self.prev_weights.get(a, 0.0)) for a in self.assets)

        commissions = sum(f.commission for f in fills.values())
        slippage = sum(f.slippage for f in fills.values())
        costs = commissions + slippage

        # PnL: naive mark-to-market on weights * price change (mocked here)
        pnl = np.random.normal(0, 0.001) * self.equity  # stub: ~0.1% dailyized noise
        self.equity += pnl - costs
        self.highwater = max(self.highwater, self.equity)
        dd = max(0.0, (self.highwater - self.equity) / max(self.highwater, 1e-9))

        self.prev_weights = cur_w
        kill = dd > self.dd_kill
        return ExecResult(pnl=float(pnl / self.equity), costs=float(costs / self.equity),
                          turnover_l1=float(turnover), drawdown=float(dd), kill=kill)
