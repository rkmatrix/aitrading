from abc import ABC, abstractmethod
from typing import Optional
from ai.execution.order_types import OrderRequest, OrderResult

class VenueAdapter(ABC):
    name: str

    @abstractmethod
    def ping(self, timeout_ms: int = 800) -> float | None:
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> dict | None:
        ...

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def fees_bps(self, symbol: str, side: str) -> float:
        ...

    def reliability(self) -> float:
        return 0.99
