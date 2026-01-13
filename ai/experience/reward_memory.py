# ai/experience/reward_memory.py
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque

class RewardMemory:
    """
    Tracks rolling PnL statistics, drawdown, costs, exposure, etc. and emits a composite reward.
    """
    def __init__(self, weights: dict, pnl_norm_window: int = 2000, dd_window: int = 1000, clip=None):
        self.w = weights
        self.pnl_window = pnl_norm_window
        self.dd_window = dd_window
        self.clip = clip

        self.pnl_hist = deque(maxlen=self.pnl_window)
        self.equity_hist = deque(maxlen=max(self.pnl_window, self.dd_window))
        self.last_equity = 0.0

    def step(self, realized_pnl: float, unrealized_pnl: float, tx_cost: float, exposure: float) -> float:
        """
        Inputs:
          realized_pnl: PnL from fills at this step
          unrealized_pnl: mark-to-market change
          tx_cost: commissions + slippage (positive number)
          exposure: e.g., sum(abs(position_value)) / equity or gross leverage proxy
        """
        # Equity change approximation
        delta = realized_pnl + unrealized_pnl - tx_cost
        equity = (self.last_equity + delta)
        self.last_equity = equity

        self.pnl_hist.append(delta)
        self.equity_hist.append(equity)

        # Normalized PnL term (z-score)
        pnl_term = 0.0
        if len(self.pnl_hist) >= 30:
            arr = np.array(self.pnl_hist, dtype=np.float32)
            mean = arr.mean()
            std = arr.std() + 1e-8
            pnl_term = (delta - mean) / std
        else:
            pnl_term = delta * 1e-3  # small scaling warm-up

        # Drawdown term (risk penalty)
        risk_term = 0.0
        if len(self.equity_hist) >= 2:
            eq = np.array(self.equity_hist, dtype=np.float32)
            running_max = np.maximum.accumulate(eq)
            dd = (running_max - eq)
            risk_term = -dd[-1] / (abs(running_max[-1]) + 1e-8)  # negative when drawdown increases

        cost_term = -abs(tx_cost)
        pos_term = -abs(exposure)

        r = (
            self.w.get("pnl", 1.0) * pnl_term
            + self.w.get("risk", -0.2) * risk_term
            + self.w.get("cost", -0.05) * cost_term
            + self.w.get("position", -0.01) * pos_term
        )

        if self.clip is not None:
            lo, hi = float(self.clip[0]), float(self.clip[1])
            r = max(lo, min(hi, r))
        return float(r)
