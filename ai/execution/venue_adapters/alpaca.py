import os, time, random
from ai.execution.venue_adapters.base import VenueAdapter
from ai.execution.order_types import OrderRequest, OrderResult

class AlpacaAdapter(VenueAdapter):
    name = "alpaca"

    def __init__(self, dry: bool = False):
        self.key = os.getenv("ALPACA_API_KEY", "")
        self.secret = os.getenv("ALPACA_API_SECRET", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.dry = dry or (os.getenv("DRY_RUN","false").lower() == "true")

    def ping(self, timeout_ms: int = 800) -> float | None:
        t0 = time.perf_counter()
        time.sleep(min(0.01, timeout_ms/1000.0))  # simulate
        return (time.perf_counter() - t0) * 1000.0

    def get_quote(self, symbol: str) -> dict | None:
        # NOTE: wire to your live quote source. placeholder:
        return {"bid": 100.00, "ask": 100.05, "last": 100.02, "size": 100}

    def place_order(self, req: OrderRequest) -> OrderResult:
        if self.dry:
            oid = f"DRY_{int(time.time()*1000)}"
            px = req.limit_price or self.get_quote(req.symbol)["last"]
            return OrderResult(ok=True, order_id=oid, status="filled",
                               venue=self.name, filled_qty=req.qty, avg_fill_price=px)
        # TODO: integrate real Alpaca REST
        return OrderResult(ok=False, order_id=None, status="rejected",
                           venue=self.name, error="Live trading not wired")

    def fees_bps(self, symbol: str, side: str) -> float:
        return 0.2  # example
