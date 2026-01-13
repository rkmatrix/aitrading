from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from time import time
from ai.execution.order_schema import Order, OrderResult
from ai.execution.brokers.base import Broker

class DryRunBroker(Broker):
    """
    Simulated broker for safe testing. Tracks positions in a JSON cache.
    """

    def __init__(self, name: str = "dry", params: Dict[str, Any] | None = None, cache_path: str | None = None):
        super().__init__(name, params)
        self.cache_path = cache_path or "data/runtime/dry_positions.json"
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._positions = self._load_positions()

    def _load_positions(self) -> Dict[str, float]:
        p = Path(self.cache_path)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}

    def _save_positions(self) -> None:
        Path(self.cache_path).write_text(json.dumps(self._positions, indent=2))

    def preflight(self) -> tuple[bool, str]:
        return True, "dry ok"

    def get_positions(self, symbols: List[str]) -> Dict[str, float]:
        return {s: float(self._positions.get(s, 0.0)) for s in symbols}

    def place_order(self, order: Order) -> OrderResult:
        # Simulate immediate fill at a synthetic price
        price_mode = self.params.get("fill_price_mode", "mid")
        px = 100.0 if price_mode == "close" else 100.1 if price_mode == "last" else 100.05

        signed_qty = order.qty if order.side.upper() == "BUY" else -order.qty
        self._positions[order.symbol] = float(self._positions.get(order.symbol, 0.0) + signed_qty)
        self._save_positions()

        oid = f"DRY_{int(time()*1000)}"
        return OrderResult(
            ok=True,
            broker=self.name,
            order_id=oid,
            status="filled",
            filled_qty=float(order.qty),
            filled_avg_price=px,
            raw={"simulated": True, "price_mode": price_mode},
        )
