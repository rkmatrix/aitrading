from __future__ import annotations
from typing import Dict, Any, List

class SimpleZScoreAnomaly:
    def __init__(self, threshold: float = 3.0, window: int = 50):
        self.threshold = threshold
        self.window = window
        self.data: List[float] = []

    def add(self, x: float) -> Dict[str, Any]:
        self.data.append(x)
        self.data = self.data[-self.window:]
        if len(self.data) < 10:
            return {"anomaly": False}
        mean = sum(self.data) / len(self.data)
        var = sum((v - mean) ** 2 for v in self.data) / len(self.data)
        std = var ** 0.5
        z = 0 if std == 0 else (x - mean) / std
        return {"anomaly": abs(z) >= self.threshold, "z": z, "mean": mean, "std": std}
