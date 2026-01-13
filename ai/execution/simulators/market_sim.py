import numpy as np

class MicroMarketSim:
    """Very light micro-sim with stochastic spread, mid, and fill model."""
    def __init__(self, seed:int=42):
        self.rng = np.random.default_rng(seed)
        self.mid = 100.0
        self.spread = 0.01
        self.vol = 0.002

    def snapshot(self):
        return {
            "mid": self.mid,
            "spread": self.spread,
            "imbalance": self.rng.normal(0, 0.2),
            "volatility": self.vol,
            "arrival_price": self.mid,
            "latency_ms": self.rng.normal(25, 10),
            "participation": 0.1 + abs(self.rng.normal(0, 0.05)),
        }

    def execute(self, qty:int, aggression:int):
        slip = self.rng.normal(0, self.vol * self.mid)
        px_adj = [ -0.5, 0.0, +0.5 ][aggression] * self.spread
        fill_px = self.mid + px_adj + slip
        fill_ratio = min(1.0, 0.3 + aggression*0.3 + abs(self.rng.normal(0.0, 0.1)))
        fill_qty = int(qty * max(0.0, min(fill_ratio, 1.0)))
        # evolve mid/spread
        self.mid += self.rng.normal(0, self.vol * self.mid * 0.1)
        self.spread = max(0.001, self.spread + self.rng.normal(0, 0.001))
        return float(fill_px), int(fill_qty), self.snapshot()
