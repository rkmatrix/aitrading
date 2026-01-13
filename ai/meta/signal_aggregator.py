from __future__ import annotations
from typing import Dict
import numpy as np
from ai.utils.config import AppConfig

class SignalAggregator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.clip = float(cfg.y("aggregator.clip", 3.0))

    def fit(self, data):
        return None

    def combine(self, signals: Dict[str, float]) -> Dict[str, float]:
        if not signals:
            return {}
        vals = np.array(list(signals.values()), dtype=float)
        mu, sd = float(vals.mean()), float(vals.std() or 1.0)
        out = {}
        for k, v in signals.items():
            z = (float(v) - mu) / sd
            z = max(min(z, self.clip), -self.clip)
            out[k] = z
        return out
