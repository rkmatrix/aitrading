from ai.execution.venue_adapters.base import VenueAdapter
from ai.execution.order_types import OrderRequest, OrderResult

class IBKRAdapter(VenueAdapter):
    name = "ibkr"

    def ping(self, timeout_ms: int = 800) -> float | None:
        return 12.0

    def get_quote(self, symbol: str) -> dict | None:
        return {"bid": 99.99, "ask": 100.02, "last": 100.00, "size": 500}

    def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(ok=False, order_id=None, status="rejected",
                           venue=self.name, error="Adapter not implemented")

    def fees_bps(self, symbol: str, side: str) -> float:
        return 0.15
