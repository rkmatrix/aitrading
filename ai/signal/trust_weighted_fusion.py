# ai/signal/trust_weighted_fusion.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _softmax(weights: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    # temperature < 1 => sharper, > 1 => flatter
    temperature = max(1e-6, float(temperature))
    keys = list(weights.keys())
    vals = [weights[k] / temperature for k in keys]
    m = max(vals) if vals else 0.0
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) if exps else 1.0
    return {k: e / s for k, e in zip(keys, exps)}


@dataclass
class FusionResult:
    fused: float
    final_weights: Dict[str, float]
    meta: Dict[str, Any]


class TrustWeightedFusion:
    """
    Phase D-6: Trust-weighted fusion.

    Inputs:
      - agent_scores: per-agent signal in [-1,1] (or any bounded small float)
      - base_weights: your existing static weights
      - trust: 0..1 (from ModelSkillMemory)
      - per-agent confidence is derived from |score| by default (safe)
    """

    def __init__(
        self,
        *,
        min_agent_conf: float = 0.15,
        max_agent_conf: float = 1.0,
        low_trust_temp: float = 1.35,   # flatter weights when trust is low
        high_trust_temp: float = 0.75,  # sharper weights when trust is high
        conf_power_low_trust: float = 0.60,
        conf_power_high_trust: float = 1.40,
    ) -> None:
        self.min_agent_conf = float(min_agent_conf)
        self.max_agent_conf = float(max_agent_conf)
        self.low_trust_temp = float(low_trust_temp)
        self.high_trust_temp = float(high_trust_temp)
        self.conf_power_low_trust = float(conf_power_low_trust)
        self.conf_power_high_trust = float(conf_power_high_trust)

    def fuse(
        self,
        *,
        agent_scores: Dict[str, float],
        base_weights: Dict[str, float],
        trust: float,
        agent_confidence: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        trust = _clip(trust, 0.0, 1.0)

        # Choose how sharp weights become as trust rises
        temperature = self.low_trust_temp + (self.high_trust_temp - self.low_trust_temp) * trust
        conf_power = self.conf_power_low_trust + (self.conf_power_high_trust - self.conf_power_low_trust) * trust

        # Default per-agent confidence: abs(score) mapped into [min,max]
        conf_map: Dict[str, float] = {}
        for k, s in agent_scores.items():
            if agent_confidence and k in agent_confidence:
                c = _safe_float(agent_confidence[k], default=0.5)
            else:
                c = abs(_safe_float(s, default=0.0))
            # map small -> min, large -> max
            c = _clip(c, 0.0, 1.0)
            c = self.min_agent_conf + (self.max_agent_conf - self.min_agent_conf) * c
            conf_map[k] = _clip(c, 0.01, 1.0)

        # Raw dynamic weights = base_weight * (confidence^power)
        dyn = {}
        for k, bw in base_weights.items():
            bw = max(0.0, _safe_float(bw, default=0.0))
            c = conf_map.get(k, 0.5)
            dyn[k] = bw * (c ** conf_power)

        # Normalize via softmax to avoid “dead” agents
        final_w = _softmax(dyn, temperature=temperature)

        # Fused score = sum(final_w[k] * score[k])
        fused = 0.0
        for k, w in final_w.items():
            fused += w * _safe_float(agent_scores.get(k, 0.0), default=0.0)

        # Keep fused bounded
        fused = _clip(fused, -1.0, 1.0)

        meta = {
            "trust": trust,
            "temperature": temperature,
            "conf_power": conf_power,
            "conf_map": conf_map,
            "dyn_weights_pre_softmax": dyn,
        }

        return FusionResult(fused=fused, final_weights=final_w, meta=meta)
