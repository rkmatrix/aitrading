# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class AllocationPolicy(ABC):
    """Interface for allocation policies."""

    @abstractmethod
    def propose_weights(
        self,
        *,
        symbols: List[str],
        signals: Dict[str, Any],
        balances: Dict[str, Any],
        positions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return target portfolio weights per symbol (sum abs(weights) <= ~1)."""
        ...
