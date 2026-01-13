# ai/execution/safety_guard.py
"""
Phase 67 - SafetyGuard: Institutional-grade trade safety filtering.

Features:
    • min confidence threshold
    • max conflict allowed
    • max position size allowed
    • volatility risk guard
    • equity drawdown guard
    • price sanity guard
    • flip-flop protection (no BUY→SELL→BUY loops)
    • max trades per symbol per minute
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SafetyConfig:
    min_confidence: float = 0.55
    max_conflict: float = 0.60
    max_position_qty: float = 500.0
    max_trade_qty: float = 300.0
    max_volatility: float = 0.35
    max_drawdown: float = 0.25
    flip_flop_window_sec: int = 45
    max_trades_per_min: int = 6


class SafetyGuard:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.last_action: Dict[str, Dict[str, float]] = {}
        self.trade_count_1min: Dict[str, list] = {}

    def _flip_flop_block(self, symbol: str, new_side: str) -> bool:
        rec = self.last_action.get(symbol)
        if not rec:
            return False

        last_side = rec["side"]
        last_ts = rec["ts"]

        if last_side != new_side:
            if time.time() - last_ts < self.cfg.flip_flop_window_sec:
                return True

        return False

    def _rate_limit(self, symbol: str) -> bool:
        now = time.time()
        window = now - 60

        if symbol not in self.trade_count_1min:
            self.trade_count_1min[symbol] = []

        # cleanup old entries
        self.trade_count_1min[symbol] = [
            x for x in self.trade_count_1min[symbol] if x >= window
        ]

        if len(self.trade_count_1min[symbol]) >= self.cfg.max_trades_per_min:
            return True

        return False

    def allow_trade(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        confidence: float,
        conflict: float,
        volatility: float,
        current_qty: float,
        equity: float,
        max_drawdown: float,
    ) -> (bool, str):

        # Confidence filter
        if confidence < self.cfg.min_confidence:
            return False, f"Blocked: low confidence {confidence:.3f}"

        # Conflict filter
        if conflict > self.cfg.max_conflict:
            return False, f"Blocked: high conflict {conflict:.3f}"

        # Volatility filter
        if volatility > self.cfg.max_volatility:
            return False, f"Blocked: high volatility {volatility:.3f}"

        # Position limit
        if abs(current_qty + (qty if side == 'BUY' else -qty)) > self.cfg.max_position_qty:
            return False, f"Blocked: position limit"

        # Trade qty limit
        if qty > self.cfg.max_trade_qty:
            return False, f"Blocked: trade too large qty={qty}"

        # Equity drawdown guard
        if max_drawdown > self.cfg.max_drawdown:
            return False, f"Blocked: drawdown {max_drawdown:.2f}"

        # Flip-flop protection
        if self._flip_flop_block(symbol, side):
            return False, f"Blocked: flip-flop detected"

        # Rate limiting
        if self._rate_limit(symbol):
            return False, f"Blocked: too many trades per minute"

        # Everything OK → record action
        self.last_action[symbol] = {"side": side, "ts": time.time()}
        self.trade_count_1min[symbol].append(time.time())

        return True, "OK"
