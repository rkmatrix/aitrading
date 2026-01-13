"""
ai/policy/agents/momentum_agent.py
----------------------------------
Momentum / trend-following agent.

Assumes obs contains a dict:
    obs["returns_window"][symbol] -> list/array of recent returns
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MomentumAgent:
    def __init__(
        self,
        *,
        symbols,
        lookback_bars: int = 20,
        name: str = "momentum",
    ) -> None:
        self.symbols = list(symbols)
        self.lookback_bars = lookback_bars
        self.name = name

    def decide(
        self,
        obs: Dict[str, Any],
        *,
        portfolio: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        returns_window: Dict[str, Any] = obs.get("returns_window", {})
        scores: Dict[str, float] = {}

        for sym in self.symbols:
            series = np.asarray(returns_window.get(sym, []), dtype=float)
            if series.size == 0:
                scores[sym] = 0.0
                continue
            tail = series[-self.lookback_bars :]
            scores[sym] = float(tail.mean())

        # normalize scores to [-1, 1] for action
        vals = np.array(list(scores.values()), dtype=float)
        if np.allclose(vals, 0.0):
            action = {sym: 0.0 for sym in self.symbols}
            conf = 0.1
        else:
            max_abs = np.max(np.abs(vals)) or 1.0
            action = {
                sym: float(scores[sym] / max_abs) for sym in self.symbols
            }
            conf = float(min(1.0, np.mean(np.abs(vals)) * 10.0))

        return {
            "action": action,
            "confidence": conf,
            "risk_score": 0.5,  # can be whipsawed in choppy markets
            "meta": {"raw_scores": scores},
        }
