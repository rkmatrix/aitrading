from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import datetime as dt


@dataclass
class Position:
    """Tracks one symbol's position state."""
    qty: float = 0.0                 # >0 long, <0 short
    avg_price: float = 0.0           # volume-weighted average entry price
    opened_at: Optional[dt.datetime] = None  # when position moved from 0 -> non-zero


class TradeLedger:
    """
    Minimal in-memory ledger for per-symbol positions.
    - Maintains qty, avg_price, opened_at
    - On fills, updates position
    - When a position returns to 0, returns a closure dict with PnL and duration info
    """

    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def _get(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position()
        return self.positions[symbol]

    @staticmethod
    def _side_to_sign(side: str) -> int:
        s = side.lower()
        if s == "buy":
            return +1
        if s == "sell":
            return -1
        raise ValueError(f"Unknown side: {side}")

    def apply_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        ts: Optional[dt.datetime] = None,
    ) -> Optional[dict]:
        """
        Apply a fill and mutate internal position.
        Returns a dict when the trade fully closes the position, otherwise None.

        Returned dict keys:
            - symbol, pnl, entry_price, exit_price, opened_at, closed_at, duration_seconds
        """
        ts = ts or dt.datetime.now()
        pos = self._get(symbol)
        sign = self._side_to_sign(side)
        trade_qty = sign * abs(qty)  # signed quantity change

        # If currently flat -> opening a new position
        if abs(pos.qty) < 1e-12:
            pos.qty = trade_qty
            pos.avg_price = float(price)
            pos.opened_at = ts
            return None

        # If adding to same direction (e.g., increasing a long or short)
        if (pos.qty > 0 and trade_qty > 0) or (pos.qty < 0 and trade_qty < 0):
            # Weighted average price update
            new_qty_abs = abs(pos.qty) + abs(trade_qty)
            if new_qty_abs < 1e-12:
                # Shouldn't happen here, but guard
                pos.qty = 0.0
                pos.avg_price = 0.0
                pos.opened_at = None
                return None
            pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(trade_qty) * float(price)) / new_qty_abs
            pos.qty += trade_qty
            # Keep original opened_at
            return None

        # Opposite direction -> reducing or flipping
        # Determine how much is closing the existing leg
        reduce_qty = min(abs(pos.qty), abs(trade_qty))
        remaining_after = pos.qty + trade_qty  # signed

        # Realized PnL for the closed portion
        if pos.qty > 0:
            # Closing long at 'price'
            pnl = (float(price) - pos.avg_price) * reduce_qty
            entry_price = pos.avg_price
            exit_price = float(price)
        else:
            # Closing short at 'price'
            pnl = (pos.avg_price - float(price)) * reduce_qty
            entry_price = pos.avg_price
            exit_price = float(price)

        # Update position qty
        pos.qty = remaining_after

        # If flat after this trade -> position closed
        if abs(pos.qty) < 1e-12:
            opened_at = pos.opened_at or ts
            closed_at = ts
            duration_seconds = int((closed_at - opened_at).total_seconds())
            # Reset state
            pos.qty = 0.0
            pos.avg_price = 0.0
            pos.opened_at = None

            return {
                "symbol": symbol,
                "pnl": pnl,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "opened_at": opened_at,
                "closed_at": closed_at,
                "duration_seconds": duration_seconds,
            }

        # If flipped to the other side, we need to set a new avg and opened_at
        if (pos.qty > 0 and remaining_after > 0) or (pos.qty < 0 and remaining_after < 0):
            # Still same side after partial reduce; avg already correct
            return None
        else:
            # Flipped — remaining_after keeps sign of new side and its magnitude is the net leftover
            pos.avg_price = float(price)  # new entry at this trade price
            pos.opened_at = ts
            return None
