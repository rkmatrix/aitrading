from __future__ import annotations
import math
from typing import Dict, List, Sequence, Optional

class FeatureEncoder:
    """
    Encodes market state dicts from StateObserver into numeric observation vectors.
    - Applies per-feature clipping (ranges)
    - Applies scaling ("minmax", "standard", "none")
    - Enforces feature order
    """

    def __init__(self, schema: Dict, fill_value: float = 0.0):
        cfg = schema or {}
        feats = (cfg.get("features") or {})
        self.order: List[str] = list((feats.get("order") or []))
        self.ranges: Dict[str, Sequence[float]] = feats.get("ranges") or {}
        self.scales: Dict[str, str] = feats.get("scales") or {}
        self.stats: Dict[str, Dict[str, float]] = feats.get("stats") or {}
        self.fill_value = cfg.get("fill_value", fill_value)

    # ---------- helpers ----------
    def _clip(self, name: str, v: float) -> float:
        r = self.ranges.get(name)
        if not r or len(r) != 2: 
            return v
        lo, hi = r
        if v < lo: return lo
        if v > hi: return hi
        return v

    def _scale(self, name: str, v: float) -> float:
        mode = self.scales.get(name, "none")
        if mode == "minmax":
            r = self.ranges.get(name)
            if not r: 
                return v
            lo, hi = r
            span = hi - lo if hi != lo else 1.0
            return (v - lo) / span
        if mode == "standard":
            st = self.stats.get(name, {})
            mean = st.get("mean", 0.0)
            std = st.get("std", 1.0) or 1.0
            return (v - mean) / std
        return v

    # ---------- main ----------
    def encode(self, state: Dict) -> List[float]:
        vec: List[float] = []
        for name in self.order:
            raw = state.get(name, self.fill_value)
            # guard non-finite
            if raw is None or (isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw))):
                raw = self.fill_value
            v = self._scale(name, self._clip(name, float(raw)))
            vec.append(float(v))
        return vec

    def encode_batch(self, states: List[Dict]) -> List[List[float]]:
        return [self.encode(s) for s in states]
