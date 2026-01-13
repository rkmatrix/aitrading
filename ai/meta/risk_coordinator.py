from __future__ import annotations
from typing import Dict
import math
from ai.utils.config import AppConfig

class RiskCoordinator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.max_w = float(cfg.y("risk.max_weight", 0.25))

    def fit(self, data):
        return None

    def target(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        exps = {k: math.exp(v) for k, v in scores.items()}
        s = sum(exps.values()) or 1.0
        base = {k: v / s for k, v in exps.items()}
        clipped = {k: min(self.max_w, v) for k, v in base.items()}
        s2 = sum(clipped.values()) or 1.0
        return {k: v / s2 for k, v in clipped.items()}
