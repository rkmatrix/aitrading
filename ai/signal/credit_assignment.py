# ai/signal/credit_assignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _safe(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

@dataclass
class CreditSnapshot:
    symbol: str
    side: str                      # "BUY" or "SELL"
    ts: float
    weights: Dict[str, float]      # dynamic fusion weights used
    scores: Dict[str, float]       # per-agent scores at entry
    fused: float                   # fused score at entry
    confidence: float              # confidence at entry (D-2)

class CreditAssigner:
    """
    Phase D-7:
    - Builds per-agent credit weights for a trade outcome.
    - Credits agents that (a) had weight and (b) aligned with trade direction.
    """

    def compute_agent_credit(self, snap: CreditSnapshot) -> Dict[str, float]:
        side = (snap.side or "").upper().strip()
        if side not in ("BUY", "SELL"):
            side = "BUY" if snap.fused >= 0 else "SELL"

        # alignment: score sign matches trade direction
        def aligned(score: float) -> bool:
            return score >= 0 if side == "BUY" else score <= 0

        raw: Dict[str, float] = {}
        for k, w in (snap.weights or {}).items():
            w = max(0.0, _safe(w, 0.0))
            s = _safe((snap.scores or {}).get(k, 0.0), 0.0)
            if aligned(s) and abs(s) > 0:
                # stronger score and higher fusion weight => more credit
                raw[k] = w * abs(s)
            else:
                raw[k] = 0.0

        tot = sum(raw.values())
        if tot <= 1e-12:
            # fallback: if nobody aligned, distribute by weights only (prevents dead agents)
            raw = {k: max(0.0, _safe(w, 0.0)) for k, w in (snap.weights or {}).items()}
            tot = sum(raw.values())

        if tot <= 1e-12:
            return {}

        return {k: _clip(v / tot, 0.0, 1.0) for k, v in raw.items()}
