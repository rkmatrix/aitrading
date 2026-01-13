from .base import CostModel, FillCost

class EmpiricalCost(CostModel):
    """Pluggable coefficients learned from fills/backtests."""
    def __init__(self, coef: dict):
        self.c = coef

    def estimate(self, *, side: int, qty: int, price: float, spread: float, participation: float, volatility: float, latency_ms: float) -> FillCost:
        fees = self.c.get("fee_per_share", 0.0) * qty
        spread_cost = self.c.get("spread_mult", 1.0) * spread * qty
        temp_impact = self.c.get("k", 0.1) * (qty ** self.c.get("alpha", 0.5)) * participation
        perm_impact = self.c.get("gamma", 0.05) * qty
        latency_slip = (latency_ms / 1000.0) * volatility * price * self.c.get("lat_mult", 0.1)
        return FillCost(fees, spread_cost, temp_impact, perm_impact, latency_slip)
