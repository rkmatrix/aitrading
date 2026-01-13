# ai/strategies/dumb_trend_revert.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import tanh
from typing import Sequence, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DumbStrategyConfig:
    trend_window_short: int = 5
    trend_window_long: int = 20
    revert_window: int = 10
    scale: float = 20.0  # scaling before tanh


class DumbStrategyEngine:
    """
    Phase 121 – Very simple "dumb" strategies:
      - Trend-follow: is price trending up/down?
      - Mean-revert: is price stretched away from recent mean?

    Outputs scores in [-1, 1].
    """

    def __init__(self, cfg: DumbStrategyConfig | None = None) -> None:
        self.cfg = cfg or DumbStrategyConfig()

    def compute_scores(self, prices: Sequence[float]) -> Dict[str, float]:
        """
        prices: sequence of recent closes (oldest → newest)
        """
        if len(prices) < min(self.cfg.trend_window_long, self.cfg.revert_window, 3):
            return {"trend_score": 0.0, "revert_score": 0.0}

        arr = np.asarray(prices, dtype=float)

        # --- Trend-follow: short vs long moving average ---
        try:
            sw = self.cfg.trend_window_short
            lw = self.cfg.trend_window_long

            ma_short = float(arr[-sw:].mean())
            ma_long = float(arr[-lw:].mean())
            if ma_long <= 0:
                trend_score = 0.0
            else:
                rel = (ma_short / ma_long) - 1.0
                trend_score = float(tanh(rel * self.cfg.scale))
        except Exception:
            logger.exception("DumbStrategyEngine: trend calc failed.")
            trend_score = 0.0

        # --- Mean-reversion: last price vs recent mean ---
        try:
            rw = self.cfg.revert_window
            window = arr[-rw:]
            mean = float(window.mean())
            last = float(window[-1])
            if mean <= 0:
                revert_score = 0.0
            else:
                dev = (mean - last) / mean  # positive if last < mean (value buy)
                revert_score = float(tanh(dev * self.cfg.scale))
        except Exception:
            logger.exception("DumbStrategyEngine: revert calc failed.")
            revert_score = 0.0

        return {
            "trend_score": trend_score,
            "revert_score": revert_score,
        }
