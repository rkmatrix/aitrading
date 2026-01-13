from __future__ import annotations
import os, json, urllib.request, urllib.error
from typing import Dict, Any, List
from ai.execution.order_schema import Order, OrderResult
from ai.execution.brokers.base import Broker

class AlpacaBroker(Broker):
    def __init__(self, name: str = "alpaca", params: Dict[str, Any] | None = None):
        super().__init__(name, params)
        self.base_url = self.params.get("base_url", "https://paper-api.alpaca.markets")
        self.api_key = os.getenv(self.params.get("api_key_env", "ALPACA_API_KEY"), "")
        self.api_secret = os.getenv(self.params.get("api_secret_env", "ALPACA_API_SECRET"), "")
        self.timeout = int(self.params.get("timeout_sec", 10))

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def preflight(self) -> tuple[bool, str]:
        if not self.api_key or not self.api_secret:
            return False, "missing api credentials"
        return True, "ok"

    def get_positions(self, symbols: List[str]) -> Dict[str, float]:
        try:
            req = urllib.request.Request(f"{self.base_url}/v2/positions", headers=self._headers())
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {s: 0.0 for s in symbols}
        except Exception:
            return {s: 0.0 for s in symbols}

        out: Dict[str, float] = {}
        for row in rows:
            sym = row.get("symbol")
            if sym in symbols:
                # qty is a string in Alpaca
                qty = float(row.get("qty", "0"))
                side = row.get("side", "long").lower()
                out[sym] = qty if side == "long" else -qty
        # Ensure all requested symbols present
        for s in symbols:
            out.setdefault(s, 0.0)
        return out

    def place_order(self, order: Order) -> OrderResult:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": order.symbol,
            "side": order.side.lower(),
            "type": order.order_type.lower(),  # market
            "time_in_force": order.time_in_force.lower(),  # day
            "qty": str(abs(order.qty)),
            "client_order_id": order.extra.get("client_order_id", order.tag),
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            oid = data.get("id") or data.get("client_order_id") or None
            status = data.get("status", "accepted")
            filled_qty = float(data.get("filled_qty") or 0)
            filled_avg_price = float(data.get("filled_avg_price") or 0) if data.get("filled_avg_price") else None
            return OrderResult(
                ok=True,
                broker=self.name,
                order_id=oid,
                status=status,
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
                raw=data,
            )
        except Exception as e:
            return OrderResult(ok=False, broker=self.name, order_id=None, status="error", error=str(e))
