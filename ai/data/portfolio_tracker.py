# ai/data/portfolio_tracker.py
"""
Safe dummy portfolio tracker for Phase 26–66.

Handles:
    • equity
    • running max drawdown
"""

from __future__ import annotations

import random
from typing import Dict


class PortfolioTracker:
    def __init__(self):
        self._equity = random.uniform(100000, 150000)
        self._peak = self._equity
        self._max_dd = 0.0

    def snapshot(self) -> Dict[str, float]:
        """
        Equity changes slowly to simulate portfolio PnL drift.
        """
        # small random drift
        drift = random.uniform(-150, +150)
        self._equity = max(20000, self._equity + drift)

        # max drawdown logic
        if self._equity > self._peak:
            self._peak = self._equity
        dd = 1.0 - (self._equity / self._peak)
        self._max_dd = max(self._max_dd, dd)

        return {
            "equity": float(self._equity),
            "max_drawdown": float(self._max_dd),
        }
