from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FillCost:
    fees: float
    spread_cost: float
    temp_impact: float
    perm_impact: float
    latency_slip: float

    @property
    def total(self) -> float:
        return self.fees + self.spread_cost + self.temp_impact + self.perm_impact + self.latency_slip

class CostModel(ABC):
    @abstractmethod
    def estimate(self, *, side: int, qty: int, price: float, spread: float, participation: float, volatility: float, latency_ms: float) -> FillCost:
        ...
