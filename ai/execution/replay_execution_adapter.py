# ai/execution/replay_execution_adapter.py
# Phase-G.1 — Virtual Capital Model & Fill Simulator

from __future__ import annotations
from typing import Dict, Any
import time


class ReplayExecutionAdapter:
    """
    Deterministic replay-only execution adapter.
    Mimics a broker but never places live orders.
    """

    def __init__(self, starting_equity: float = 100_000.0):
        self.starting_equity = float(starting_equity)
        self.cash = float(starting_equity)
        self.positions: Dict[str, Dict[str, float]] = {}
        self.realized_pnl = 0.0
        self.equity = float(starting_equity)

    # ------------------------------------------------------------------
    # Portfolio snapshot (matches live expectations)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "positions": self.positions,
            "realized_pnl": self.realized_pnl,
        }

    # ------------------------------------------------------------------
    # Deterministic fill (price = bar close)
    # ------------------------------------------------------------------
    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        side = side.upper()
        qty = float(qty)
        price = float(price)
        notional = qty * price

        if side == "BUY":
            if self.cash < notional:
                return {"ok": False, "reason": "insufficient_cash"}

            self.cash -= notional
            pos = self.positions.get(symbol, {"qty": 0.0, "price": 0.0})
            new_qty = pos["qty"] + qty
            avg_price = (
                (pos["qty"] * pos["price"] + notional) / new_qty
                if new_qty > 0
                else price
            )
            self.positions[symbol] = {"qty": new_qty, "price": avg_price}

        elif side == "SELL":
            pos = self.positions.get(symbol)
            if not pos or pos["qty"] < qty:
                return {"ok": False, "reason": "insufficient_position"}

            self.cash += notional
            pnl = qty * (price - pos["price"])
            self.realized_pnl += pnl
            pos["qty"] -= qty
            if pos["qty"] <= 0:
                self.positions.pop(symbol, None)

        # Recompute equity
        self._mark_to_market(price_by_symbol={symbol: price})

        return {
            "ok": True,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "ts": time.time(),
        }

    # ------------------------------------------------------------------
    # Mark-to-market equity
    # ------------------------------------------------------------------
    def _mark_to_market(self, price_by_symbol: Dict[str, float]) -> None:
        unrealized = 0.0
        for sym, pos in self.positions.items():
            px = price_by_symbol.get(sym, pos["price"])
            unrealized += pos["qty"] * (px - pos["price"])
        self.equity = self.cash + unrealized
