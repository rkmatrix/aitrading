from dataclasses import dataclass
from typing import Dict


@dataclass
class MetaParams:
    entropy_coef: float
    lambda_drawdown: float
    kappa_cost: float
    temperature: float
    lr: float
    trust_region_l1: float


class MetaController:
    """Simple heuristic regime-aware hyper-parameter tuner."""

    def select(self, regime: Dict[str, float]) -> MetaParams:
        vol = float(regime.get("vol_regime", 0.5))
        liq = float(regime.get("liquidity_regime", 0.5))
        entropy = 0.005 + 0.02 * vol
        tr = 0.10 if vol > 0.7 else (0.15 if vol > 0.4 else 0.25)
        kappa = 0.25 + 0.25 * (1 - liq)
        lam = 0.15 + 0.20 * vol
        temp = 0.8 + 0.6 * (1 - vol)
        lr = 3e-4 * (0.75 if vol > 0.7 else 1.0)
        return MetaParams(entropy, lam, kappa, temp, lr, tr)
