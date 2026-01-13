import time
from typing import Dict

class SimpleTicker:
    """Wall-clock scheduler to throttle child order emissions."""
    def __init__(self, period_sec: float):
        self.period = max(0.5, float(period_sec))
        self._next = time.time()

    def due(self) -> bool:
        now = time.time()
        if now >= self._next:
            self._next = now + self.period
            return True
        return False
