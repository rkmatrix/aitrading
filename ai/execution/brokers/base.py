from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ai.execution.order_schema import Order, OrderResult

class Broker(ABC):
    name: str

    def __init__(self, name: str, params: Dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def preflight(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    def get_positions(self, symbols: List[str]) -> Dict[str, float]:
        """Return current position sizes by symbol (signed: +long, -short)."""
        ...

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        ...
