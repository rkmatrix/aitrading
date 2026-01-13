# ai/data/position_tracker.py
"""
Safe dummy position tracker.

Tracks:
    • qty per symbol
    • updates from trades
"""

from __future__ import annotations

from typing import Dict


class PositionTracker:
    def __init__(self):
        self._pos: Dict[str, Dict[str, float]] = {}

    def get(self, symbol: str) -> Dict[str, float]:
        return self._pos.get(symbol, {"qty": 0.0})

    def update_from_trade(self, symbol: str, side: str, qty: float, price: float):
        """
        Updates positions based on executed trade.
        """
        rec = self._pos.get(symbol, {"qty": 0.0})
        if side == "BUY":
            rec["qty"] = rec.get("qty", 0.0) + qty
        elif side == "SELL":
            rec["qty"] = rec.get("qty", 0.0) - qty
        self._pos[symbol] = rec
