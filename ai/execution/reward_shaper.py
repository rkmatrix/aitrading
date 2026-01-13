# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class RewardConfig:
    slippage_weight: float = 1.0
    inventory_weight: float = 0.1
    timing_weight: float = 0.05
    penalty_violations: float = 5.0

class ExecutionReward:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(self, slippage_bps: float, inventory_left: float,
                time_penalty: float, violated: bool) -> float:
        r = - self.cfg.slippage_weight * slippage_bps
        r -= self.cfg.inventory_weight * inventory_left
        r -= self.cfg.timing_weight * time_penalty
        if violated:
            r -= self.cfg.penalty_violations
        return float(r)
