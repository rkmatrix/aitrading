# ai/memory/execution_quality_memory.py
# Phase 123 — Execution Quality Memory (EQM)
# Updated to match Phase 26 call signature and allow router usage later.

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Any


class ExecutionQualityMemory:
    """
    Tracks execution quality for each symbol and produces:
        - eq_score ∈ [0,1]
        - aggression_scale ∈ [0.5, 1.0] (for router or fusion engine)

    Phase 26 calls:
        self.eqm.record(
            symbol=symbol,
            qty=float(final_qty),
            price=float(fill_price),
            mid_price=float(mid_price),
            spread=float(spread),
        )

    So this class MUST accept those parameters.
    """

    def __init__(self, maxlen: int = 50) -> None:
        self.records = deque(maxlen=maxlen)

    # ------------------------------------------------------
    # Main recording entrypoint (used by Phase 26)
    # ------------------------------------------------------
    def record(
        self,
        *,
        symbol: str,
        qty: float,
        price: float,
        mid_price: float,
        spread: float,
    ) -> None:
        """Record slippage + spread stats for a single fill."""
        if not mid_price:
            mid_price = price

        slippage = abs(price - mid_price) / mid_price if mid_price else 0
        spread_penalty = min(spread / mid_price, 1.0) if mid_price else 0

        eq = max(0.0, 1.0 - (slippage * 0.6 + spread_penalty * 0.4))

        self.records.append(
            {
                "t": time.time(),
                "symbol": symbol,
                "qty": qty,
                "slippage": slippage,
                "spread_penalty": spread_penalty,
                "eq_score": eq,
            }
        )

    # ------------------------------------------------------
    # Optional compatibility hook for any future router usage
    # ------------------------------------------------------
    def record_metrics(
        self,
        *,
        symbol: str,
        price: float,
        mid_price: float,
        spread: float,
        qty: float,
        side: str,
    ) -> None:
        """Router-friendly wrapper around record()."""
        self.record(
            symbol=symbol,
            qty=qty,
            price=price,
            mid_price=mid_price,
            spread=spread,
        )

    # ------------------------------------------------------
    # Compute aggregate quality and aggressiveness
    # ------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """
        Returns:
            {
              "eq_score": float  in [0,1],
              "aggression_scale": float in [0.5,1.0]
            }
        """
        if not self.records:
            return {
                "eq_score": 1.0,
                "aggression_scale": 1.0,
            }

        avg = sum(r["eq_score"] for r in self.records) / len(self.records)

        # Convert eq_score → aggression clamp
        if avg < 0.55:
            scale = 0.60
        elif avg < 0.70:
            scale = 0.75
        else:
            scale = 1.00

        return {
            "eq_score": avg,
            "aggression_scale": scale,
        }
