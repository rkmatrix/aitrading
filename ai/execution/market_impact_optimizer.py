# ai/execution/market_impact_optimizer.py
import numpy as np

class MarketImpactOptimizer:
    """Estimate and adjust orders to minimize market impact."""

    def __init__(self, avg_daily_vol, impact_coefficient=0.1):
        self.adv = avg_daily_vol
        self.impact_coefficient = impact_coefficient

    def expected_impact(self, order_size):
        ratio = order_size / (self.adv + 1e-8)
        return self.impact_coefficient * np.power(ratio, 0.5)

    def adjust_price(self, base_price, order_side, order_size):
        impact = self.expected_impact(order_size)
        if order_side == "buy":
            return base_price * (1 + impact)
        else:
            return base_price * (1 - impact)
