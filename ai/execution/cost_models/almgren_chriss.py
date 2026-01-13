from .base import CostModel, FillCost

class AlmgrenChrissCost(CostModel):
    def __init__(self, fee_per_share: float, temp_k: float, perm_gamma: float, spread_mult: float = 1.0):
        self.fee = fee_per_share
        self.k = temp_k
        self.gamma = perm_gamma
        self.sp_mult = spread_mult

    def estimate(self, *, side: int, qty: int, price: float, spread: float, participation: float, volatility: float, latency_ms: float) -> FillCost:
        fees = self.fee * qty
        spread_cost = self.sp_mult * spread * qty
        temp_impact = self.k * (qty ** 0.5) * max(participation, 1e-6)
        perm_impact = self.gamma * qty
        latency_slip = (latency_ms / 1000.0) * volatility * price * 0.1
        return FillCost(fees, spread_cost, temp_impact, perm_impact, latency_slip)
