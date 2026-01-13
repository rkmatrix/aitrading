import numpy as np
from collections import deque

class MicroAlpha:
    """Simple micro alpha: short-horizon momentum + imbalance filter."""
    def __init__(self, window:int=20):
        self.q = deque(maxlen=window)

    def update(self, mid: float) -> float:
        self.q.append(mid)
        if len(self.q) < 3: return 0.0
        x = np.asarray(self.q, dtype=float)
        ret = (x[-1] - x[0]) / max(x[0], 1e-6)
        return float(ret)
