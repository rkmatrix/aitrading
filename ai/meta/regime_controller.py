# ai/meta/regime_controller.py
from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class RegimeBlend:
    bull: float
    bear: float
    sideways: float

class RegimeController:
    """
    Ultra-light controller: given the live regime & memory suggestion,
    produce a stable, smoothed blend usable by your policy selector
    (or just for logging / LR scaling).
    """
    def __init__(self, smooth: float = 0.8):
        self.smooth = float(smooth)
        self._last = RegimeBlend(1/3, 1/3, 1/3)

    def update(self, suggestion: Dict[str, float]) -> RegimeBlend:
        sug = RegimeBlend(
            bull=float(suggestion.get("bull", 1/3)),
            bear=float(suggestion.get("bear", 1/3)),
            sideways=float(suggestion.get("sideways", 1/3)),
        )
        # EMA smoothing
        def ema(prev, new): return self.smooth*prev + (1-self.smooth)*new
        self._last = RegimeBlend(
            bull=ema(self._last.bull, sug.bull),
            bear=ema(self._last.bear, sug.bear),
            sideways=ema(self._last.sideways, sug.sideways),
        )
        # normalize
        s = self._last.bull + self._last.bear + self._last.sideways
        if s <= 1e-8:
            self._last = RegimeBlend(1/3, 1/3, 1/3)
        else:
            self._last = RegimeBlend(self._last.bull/s, self._last.bear/s, self._last.sideways/s)
        return self._last

    def as_dict(self) -> Dict[str, float]:
        return {"bull": self._last.bull, "bear": self._last.bear, "sideways": self._last.sideways}
