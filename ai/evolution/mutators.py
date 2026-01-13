# ai/evolution/mutators.py
from __future__ import annotations
import math, random
from typing import Any, Dict

def _clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

def mutate_value(v: float, spec: Dict[str, Any]) -> float:
    sigma = float(spec.get("sigma", 0.2))
    low = float(spec.get("low", v*0.5))
    high = float(spec.get("high", v*1.5))
    if spec.get("log", False):
        # mutate in log space
        lv = math.log(max(v, 1e-12))
        lv += random.gauss(0.0, sigma)
        nv = math.exp(lv)
    else:
        nv = v + random.gauss(0.0, sigma*max(abs(v), 1e-6))
    return _clip(nv, low, high)

def crossover(a: float, b: float) -> float:
    # simple blend crossover
    alpha = random.random()
    return alpha*a + (1 - alpha)*b

def mutate_hparams(hp: Dict[str, Any], spec: Dict[str, Any], mutate_prob: float) -> Dict[str, Any]:
    out = dict(hp)
    for k, s in spec.items():
        if k not in out:
            continue
        if random.random() < mutate_prob:
            out[k] = mutate_value(float(out[k]), s)
    return out

def crossover_hparams(hp_a: Dict[str, Any], hp_b: Dict[str, Any], spec: Dict[str, Any], crossover_prob: float) -> Dict[str, Any]:
    out = dict(hp_a)
    for k in spec.keys():
        if k in hp_a and k in hp_b and random.random() < crossover_prob:
            out[k] = crossover(float(hp_a[k]), float(hp_b[k]))
    return out
