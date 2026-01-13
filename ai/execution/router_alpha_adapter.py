"""
ai/execution/router_alpha_adapter.py

Phase 82 — Translate Execution Alpha results into broker routing weights.

The SmartOrderRouter v3 already uses a dynamic router_scores.json.
This adapter converts:
    (symbol → side → broker → stats)
into:
    broker → score → normalized weight
"""

from __future__ import annotations
import math
from typing import Dict, Any


def softmax(scores: Dict[str, float], alpha: float = 3.0) -> Dict[str, float]:
    """
    Softmax with temperature α (alpha).
    """
    exp_vals = {k: math.exp(v * alpha) for k, v in scores.items()}
    total = sum(exp_vals.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: exp_vals[k] / total for k in scores}


def aggregate_broker_scores(alpha_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Input example:
    {
      "AAPL": {
        "BUY": {
          "alpaca":   { "score": 0.32, ... },
          "polygon":  { "score": -0.01, ... }
        },
        "SELL": {
          ...
        }
      },
      "MSFT": { ... }
    }

    We calculate per-broker avg score across all symbols/sides.
    """
    broker_scores: Dict[str, list] = {}

    for sym, side_map in alpha_dict.items():
        for side, broker_map in side_map.items():
            for broker, stats in broker_map.items():
                s = float(stats.get("score", 0.0))
                broker_scores.setdefault(broker, []).append(s)

    # Compute mean score per broker
    final_scores = {
        broker: (sum(vals) / len(vals)) if len(vals) > 0 else 0.0
        for broker, vals in broker_scores.items()
    }

    # Convert to weights
    return softmax(final_scores)
