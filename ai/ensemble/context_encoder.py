# ai/ensemble/context_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class ContextVector:
    x: np.ndarray
    fields: Dict[str, float]


class ContextEncoder:
    """Encodes market state → compact numeric context for the meta-controller."""
    def __init__(self, field_order=None):
        self.field_order = field_order or [
            "vix",
            "vol_20d",
            "trend_strength",
            "meanrev_score",
            "macro_risk",
            "sentiment",
            "corr_spy",
            "drawdown",
        ]

    def encode(self, features: Dict[str, float]) -> ContextVector:
        vals = []
        fields_out = {}
        for f in self.field_order:
            v = float(features.get(f, 0.0))
            fields_out[f] = v
            vals.append(v)
        return ContextVector(x=np.array(vals, dtype=float), fields=fields_out)
