import time
from ai.execution.venue_adapters.base import VenueAdapter
from ai.execution.order_types import OrderRequest, OrderResult

class PaperAdapter(VenueAdapter):
    name = "paper"

    def ping(self, timeout_ms: int = 800) -> float | None:
        t0 = time.perf_counter()
        time.sleep(0.005)
        return (time.perf_counter() - t0) * 1000.0

    def get_quote(self, symbol: str) -> dict | None:
        return {"bid": 100.00, "ask": 100.1, "last": 100.05, "size": 50}

    def place_order(self, req: OrderRequest) -> OrderResult:
        oid = f"PAPER_{int(time.time()*1000)}"
        px = req.limit_price or self.get_quote(req.symbol)["last"]
        return OrderResult(ok=True, order_id=oid, status="filled",
                           venue=self.name, filled_qty=req.qty, avg_fill_price=px)

    def fees_bps(self, symbol: str, side: str) -> float:
        return 0.0
