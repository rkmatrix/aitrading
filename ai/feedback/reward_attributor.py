# ai/feedback/reward_attributor.py
from __future__ import annotations
import math
from typing import Dict, Any

def compute_returns(prev_px: Dict[str, float], curr_px: Dict[str, float]) -> Dict[str, float]:
    """Simple one-step arithmetic return per symbol."""
    ret: Dict[str, float] = {}
    for s, p0 in prev_px.items():
        p1 = float(curr_px.get(s, p0))
        if p0 > 0:
            ret[s] = (p1 - p0) / p0
        else:
            ret[s] = 0.0
    return ret

def dot_attribution(predictions: Dict[str, Dict[str, Any]],
                    rets: Dict[str, float],
                    confidence_power: float = 1.0,
                    scale: float = 1.0,
                    clip: float = 1.0) -> Dict[str, float]:
    """
    Reward_i = scale * confidence_i^p * sum_s ( target_i[s] * return_s )
    Optionally clip to [-clip, +clip] if clip>0
    """
    out: Dict[str, float] = {}
    for name, pred in predictions.items():
        targets = pred.get("targets", {}) or {}
        conf = float(pred.get("confidence", 1.0))
        eff = conf ** confidence_power
        r = 0.0
        for s, t in targets.items():
            r += float(t) * float(rets.get(s, 0.0))
        r *= eff * scale
        if clip and clip > 0:
            r = max(min(r, clip), -clip)
        out[name] = r
    return out
