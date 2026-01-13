import random, time
from ai.execution.venue_adapters.base import VenueAdapter
from ai.execution.order_types import OrderRequest, OrderResult

class MockAdapter(VenueAdapter):
    name = "mock"

    def ping(self, timeout_ms: int = 800) -> float | None:
        return random.uniform(3, 25)

    def get_quote(self, symbol: str) -> dict | None:
        px = 100.0 + random.uniform(-0.1, 0.1)
        return {"bid": round(px-0.03, 2), "ask": round(px+0.03, 2), "last": round(px,2), "size": 100}

    def place_order(self, req: OrderRequest) -> OrderResult:
        time.sleep(random.uniform(0.001, 0.005))
        px = req.limit_price or self.get_quote(req.symbol)["last"]
        return OrderResult(ok=True, order_id=f"MOCK_{int(time.time()*1000)}",
                           status="filled", venue=self.name, filled_qty=req.qty, avg_fill_price=px)

    def fees_bps(self, symbol: str, side: str) -> float:
        return 0.1
