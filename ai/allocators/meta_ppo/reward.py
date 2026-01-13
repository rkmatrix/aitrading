from dataclasses import dataclass
from math import log


@dataclass
class RewardConfig:
    mode: str = "pnl_risk"
    kappa_cost: float = 0.35
    lambda_drawdown: float = 0.20
    psi_turnover: float = 0.05


class RewardEngine:
    """Compute risk-aware reward."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(self, pnl: float, costs: float, drawdown: float, turnover_l1: float) -> float:
        if self.cfg.mode == "utility":
            r = log(1.0 + max(-0.99, pnl - costs))
        else:
            r = pnl - costs
        r -= self.cfg.lambda_drawdown * max(0.0, drawdown)
        r -= self.cfg.psi_turnover * max(0.0, turnover_l1)
        return float(r)
