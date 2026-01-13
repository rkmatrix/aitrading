import uuid
from typing import Dict, Any
from ai.execution.order_types import OrderRequest, OrderResult
from ai.execution.venue_adapters.base import VenueAdapter
from tools.latency_probe import measure_latencies
from tools.fee_tables import default_equity_fees_bps
from ai.utils.log_utils import get_logger

class SmartOrderRouter:
    def __init__(self, cfg: Dict[str, Any], adapters: Dict[str, VenueAdapter]):
        self.cfg = cfg
        self.adapters = adapters
        self.weights = cfg["router"]["score_weights"]
        self.logger = get_logger("SmartOrderRouter")

    def _venue_score(self, venue: str, quote: dict, latency_ms: float) -> float:
        w = self.weights
        # crude proxies
        spread = (quote["ask"] - quote["bid"]) / ((quote["ask"] + quote["bid"])/2)
        fill_quality = 1.0 - min(max(spread * 10000 / 25.0, 0.0), 1.0)  # 0..1
        fees = default_equity_fees_bps(venue)
        fee_score = 1.0 - min(fees / 50.0, 1.0)                          # 0..1 against 50 bps cap
        liq = min(quote.get("size", 100) / 1000.0, 1.0)                   # 0..1
        lat_score = 1.0 - min(latency_ms / 250.0, 1.0)                    # 0..1
        reliability = self.adapters[venue].reliability()

        score = (
            w["latency"] * lat_score +
            w["fill_quality"] * fill_quality +
            w["fees"] * fee_score +
            w["liquidity"] * liq +
            w["reliability"] * reliability
        )
        return float(score)

    def select_venue(self, symbol: str) -> str | None:
        timeout = int(self.cfg["latency"]["ping_timeout_ms"])
        lats = measure_latencies(self.adapters, timeout)
        candidates = []
        for name, ad in self.adapters.items():
            q = ad.get_quote(symbol)
            if not q: continue
            lat = lats.get(name, 999.0)
            sc = self._venue_score(name, q, lat)
            candidates.append((sc, name, q, lat))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        best_score, best_name, _, _ = candidates[0]
        if best_score < float(self.cfg["router"]["min_venue_score"]):
            return None
        return best_name

    def route(self, req: OrderRequest) -> OrderResult:
        venue_name = self.select_venue(req.symbol) or self.cfg["venues"].get("failover")
        if not venue_name:
            return OrderResult(ok=False, order_id=None, status="rejected", venue="none", error="no_venue")
        adapter = self.adapters[venue_name]
        if not req.client_order_id:
            req.client_order_id = str(uuid.uuid4())[:12]
        self.logger.info("Routing %s %s@%s via %s", req.side, req.qty, req.symbol, venue_name)
        return adapter.place_order(req)
