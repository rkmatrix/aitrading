from .base import ExecutionPolicy
from typing import Dict, Any

class TWAP(ExecutionPolicy):
    def __init__(self, total_qty:int, steps:int):
        self.total = total_qty
        self.steps = max(steps, 1)
        self.sent = 0

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        remaining = max(self.total - self.sent, 0)
        size = max(remaining // (state.get("steps_left", 1)), 0)
        self.sent += size
        return {"size": size, "aggression": "passive", "price": None}

class POV(ExecutionPolicy):
    def __init__(self, participation: float = 0.1, min_size:int=100):
        self.pov = participation
        self.min = min_size

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        est_vol = max(state.get("est_volume", 0), 1)
        size = max(int(self.pov * est_vol), self.min)
        return {"size": size, "aggression": "mid", "price": None}
