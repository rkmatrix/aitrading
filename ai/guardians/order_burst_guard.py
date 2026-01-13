from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import deque

@dataclass
class OrderBurstGuard:
    max_orders_per_minute: int = 5
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=256))

    def allow(self) -> bool:
        """Simple rolling 60s window burst control."""
        now = time.time()
        # prune >60s
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_orders_per_minute:
            return False
        self._timestamps.append(now)
        return True
