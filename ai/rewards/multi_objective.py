# ai/rewards/multi_objective.py
from __future__ import annotations
import numpy as np

class MultiObjectiveReward:
    """
    Combines return, risk, turnover and explicit cost into a single scalar.
    Reward = r_t  - risk_penalty*σ_port  - turnover_penalty*turnover  - cost_penalty*cost
    """
    def __init__(
        self,
        risk_penalty: float = 4.0,
        turnover_penalty: float = 2.0,
        cost_penalty: float = 1.0,
        cost_bps: float = 1.0,
        return_scale: float = 1.0,
    ):
        self.risk_penalty = float(risk_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.cost_penalty = float(cost_penalty)
        self.cost_bps = float(cost_bps)  # per unit turnover
        self.return_scale = float(return_scale)

    def cost_from_turnover(self, turnover: float) -> float:
        # cost_bps per 1.0 of turnover (gross change in weights)
        return (self.cost_bps * 1e-4) * float(turnover)

    def compute(self, r_t: float, risk: float, turnover: float, cost: float) -> float:
        return (
            self.return_scale * float(r_t)
            - self.risk_penalty * float(risk)
            - self.turnover_penalty * float(turnover)
            - self.cost_penalty * float(cost)
        )
