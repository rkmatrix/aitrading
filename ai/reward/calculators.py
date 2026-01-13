from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .sources import Event

@dataclass
class RewardWeights:
    pnl_realized: float = 1.0
    pnl_unrealized: float = 0.0
    drawdown_penalty: float = -0.5
    slippage_penalty: float = -0.2
    risk_penalty: float = -0.3

@dataclass
class RewardParams:
    max_drawdown_window: int = 200
    risk_target_vol: float = 0.02
    slippage_bps: float = 2.0

class DrawdownTracker:
    def __init__(self, window: int = 200):
        self.window = max(2, int(window))
        self._equity_series = []

    def update(self, equity: float) -> float:
        self._equity_series.append(equity)
        if len(self._equity_series) > self.window:
            self._equity_series = self._equity_series[-self.window:]
        peak = max(self._equity_series) if self._equity_series else equity
        dd = (equity - peak) / peak if peak != 0 else 0.0
        return dd  # negative when under water

class RewardCalculator:
    def __init__(self, weights: RewardWeights, params: RewardParams):
        self.w = weights
        self.p = params
        self.dd = DrawdownTracker(window=self.p.max_drawdown_window)
        self.equity = 0.0

    def compute(self, e: Event) -> Dict:
        # Simple equity accumulation
        self.equity += e.realized_pnl
        # Unrealized PnL *not* added to equity, but included in reward
        drawdown = self.dd.update(self.equity)

        # Components
        c_real = e.realized_pnl
        c_unreal = e.unrealized_pnl
        c_slip = -abs(e.slippage)          # penalty
        risk_dev = abs(e.risk) - self.p.risk_target_vol
        c_risk = -max(0.0, risk_dev)       # penalize only when above target

        reward = (
            self.w.pnl_realized     * c_real +
            self.w.pnl_unrealized   * c_unreal +
            self.w.drawdown_penalty * abs(min(0.0, drawdown)) +
            self.w.slippage_penalty * abs(c_slip) +
            self.w.risk_penalty     * abs(c_risk)
        )

        return {
            "reward": reward,
            "components": {
                "pnl_realized": c_real,
                "pnl_unrealized": c_unreal,
                "drawdown": drawdown,
                "slippage": e.slippage,
                "risk_dev": risk_dev
            },
            "equity": self.equity
        }
