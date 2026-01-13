from __future__ import annotations
import os, time, random, threading
from typing import Dict, Any, Optional
from ai.execution.utils.order_logger import log_order
from ai.execution.utils.telegram_notifier import send_telegram_alert

# ------------------------------------------------------------------
# Global store for simulated DRY-RUN orders
_DRYRUN_ORDERS: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------
def _simulate_dryrun_fill(order_id: str):
    """Background thread that gradually fills a DRY_RUN order."""
    order = _DRYRUN_ORDERS.get(order_id)
    if not order:
        return

    latency_low, latency_high = (0.5, 2.0)
    if os.getenv("DRY_LATENCY_RANGE"):
        try:
            parts = os.getenv("DRY_LATENCY_RANGE").split(",")
            latency_low, latency_high = float(parts[0]), float(parts[1])
        except Exception:
            pass

    slip_bps = float(os.getenv("DRY_SLIPPAGE_BPS", "5"))
    allow_partials = os.getenv("DRY_PARTIALS", "true").lower() == "true"

    try:
        order["status"] = "new"
        time.sleep(random.uniform(latency_low, latency_high))

        if allow_partials and order["qty"] > 1:
            order["status"] = "partially_filled"
            partial = max(1, int(order["qty"] * random.uniform(0.3, 0.6)))
            order["filled_qty"] = partial
            slip = random.uniform(-slip_bps, slip_bps) / 1e4
            order["filled_avg_price"] = round(order["limit_price"] * (1 + slip), 2)
            log_order("dry_partial_fill", dict(order))
            time.sleep(random.uniform(latency_low, latency_high))

        order["status"] = "filled"
        order["filled_qty"] = int(order["qty"])
        slip = random.uniform(-slip_bps, slip_bps) / 1e4
        order["filled_avg_price"] = round(order["limit_price"] * (1 + slip), 2)
        log_order("dry_filled", dict(order))
    except Exception as e:
        print(f"[DRY RUN SIM ERROR] {e}")


# ------------------------------------------------------------------
class AlpacaBroker:
    """
    Alpaca REST wrapper with:
      • Flexible env detection (ALPACA_* / APCA_*)
      • Timeout & retry/backoff
      • Safe TIF fallback + price rounding
      • Realistic DRY-RUN simulation (latency/slippage/partials)
      • CSV logging
      • Telegram alerts only for buy/sell placements
    """

    def __init__(self, *, read_timeout: float = 5.0, retries: int = 3, backoff: float = 0.25):
        key = (
            os.getenv("ALPACA_API_KEY")
            or os.getenv("APCA_API_KEY_ID")
            or os.getenv("ALPACA_KEY_ID")
        )
        sec = (
            os.getenv("ALPACA_API_SECRET")
            or os.getenv("ALPACA_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
        )
        base = (
            os.getenv("ALPACA_BASE_URL")
            or os.getenv("ALPACA_API_BASE")
            or os.getenv("APCA_API_BASE_URL")
            or "https://paper-api.alpaca.markets"
        )

        if not key or not sec:
            raise EnvironmentError("Missing ALPACA_API_KEY / ALPACA_API_SECRET or APCA_API_KEY_ID / APCA_API_SECRET_KEY")

        try:
              # type: ignore
        except Exception as e:
            raise ImportError("alpaca_trade_api not installed. Run: pip install alpaca-trade-api") from e

        try:
            self.api = REST(key_id=key, secret_key=sec, base_url=base, timeout=read_timeout)
        except TypeError:
            self.api = REST(key_id=key, secret_key=sec, base_url=base)

        self.retries = retries
        self.backoff = backoff

    # ------------------------------------------------------------------
    def _try(self, fn, *args, **kwargs):
        last = None
        for i in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                time.sleep(self.backoff * (2 ** i))
        raise last  # type: ignore

    # ------------------------------------------------------------------
    def _normalize_order(self, o) -> Dict[str, Any]:
        """Normalize Alpaca order objects or dicts."""
        if hasattr(o, "_raw"):
            raw = getattr(o, "_raw", {})
        elif isinstance(o, dict):
            raw = o
        else:
            try:
                raw = vars(o)
            except Exception:
                raw = {}

        def pick(attr, default=None):
            return getattr(o, attr, None) or raw.get(attr, default)

        return {
            "order_id": pick("id"),
            "status": pick("status"),
            "filled_qty": int(float(pick("filled_qty", 0) or 0)),
            "filled_avg_price": float(pick("filled_avg_price", 0) or 0),
            "symbol": pick("symbol"),
            "side": pick("side"),
            "type": pick("type"),
            "limit_price": round(float(pick("limit_price", 0) or 0), 2) if pick("limit_price") else None,
            "time_in_force": pick("time_in_force"),
        }

    # ------------------------------------------------------------------
    def place(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        price: float | None = None,
        tif: str = "ioc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        order_type: 'market' | 'limit'
        DRY_RUN simulation enabled when DRY_RUN=true
        """
        # --- DRY-RUN simulation ---
        if os.getenv("DRY_RUN", "").lower() == "true":
            px = round(price or 100.0 + random.uniform(-0.1, 0.1), 2)
            oid = f"DRY_{int(time.time()*1000)}"
            order_data = {
                "order_id": oid,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "limit_price": px,
                "qty": qty,
                "filled_qty": 0,
                "filled_avg_price": 0.0,
                "status": "new",
                "time_in_force": tif,
            }
            _DRYRUN_ORDERS[oid] = order_data
            threading.Thread(target=_simulate_dryrun_fill, args=(oid,), daemon=True).start()
            print(f"[DRY RUN] Placed simulated {side.upper()} {qty} {symbol} @ {px} ({order_type}/{tif}) → {oid}")
            log_order("place_dryrun", order_data)
            # Telegram alert only for buy/sell orders
            if side.lower() in ("buy", "sell"):
                send_telegram_alert(f"🟢 <b>{side.upper()} {qty} {symbol}</b> placed (DRY RUN) @ {px}")
            return order_data

        # --- Real trading mode ---
        order_type = order_type.lower()
        tif = tif.lower()
        side = side.lower()
        if order_type not in ("market", "limit"):
            raise ValueError("order_type must be 'market' or 'limit'")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        valid_tif = {"day", "gtc", "opg", "cls", "ioc", "fok"}
        if tif not in valid_tif:
            tif = "day"
        if order_type == "limit" and tif == "ioc":
            tif = "day"

        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "qty": qty,
            "time_in_force": tif,
        }
        if client_order_id:
            params["client_order_id"] = client_order_id
        if order_type == "limit":
            if price is None:
                raise ValueError("limit order requires price")
            params["limit_price"] = round(float(price), 2)

        try:
            o = self._try(self.api.submit_order, **params)
        except Exception as e:
            if "time_in_force" in str(e).lower() and tif != "day":
                params["time_in_force"] = "day"
                o = self._try(self.api.submit_order, **params)
            else:
                raise

        order_data = self._normalize_order(o)
        log_order("place", order_data)
        if side.lower() in ("buy", "sell"):
            send_telegram_alert(f"🟢 <b>{side.upper()} {qty} {symbol}</b> placed @ {order_data.get('limit_price') or 'MKT'}")
        return order_data

    # ------------------------------------------------------------------
    def cancel(self, order_id: str) -> bool:
        if os.getenv("DRY_RUN", "").lower() == "true":
            print(f"[DRY RUN] Cancel order {order_id}")
            if order_id in _DRYRUN_ORDERS:
                _DRYRUN_ORDERS[order_id]["status"] = "canceled"
            log_order("cancel_dryrun", {"order_id": order_id, "status": "canceled"})
            return True

        try:
            self._try(self.api.cancel_order, order_id)
            log_order("cancel", {"order_id": order_id, "status": "canceled"})
            return True
        except Exception as e:
            log_order("cancel_failed", {"order_id": order_id, "error": str(e)})
            return False

    # ------------------------------------------------------------------
    def replace(self, order_id: str, *, limit_price: Optional[float] = None, qty: Optional[int] = None) -> Dict[str, Any]:
        if os.getenv("DRY_RUN", "").lower() == "true":
            if order_id in _DRYRUN_ORDERS:
                order = _DRYRUN_ORDERS[order_id]
                if limit_price:
                    order["limit_price"] = round(float(limit_price), 2)
                if qty:
                    order["qty"] = qty
                log_order("replace_dryrun", dict(order))
                return order
            raise ValueError(f"Unknown dry-run order_id {order_id}")

        if limit_price is not None:
            limit_price = round(float(limit_price), 2)
        try:
            o = self._try(self.api.replace_order, order_id, limit_price=limit_price, qty=qty)
            order_data = self._normalize_order(o)
            log_order("replace", order_data)
            return order_data
        except Exception as e:
            self.cancel(order_id)
            side = os.getenv("LAST_SIDE", "buy")
            symbol = os.getenv("LAST_SYMBOL", "AAPL")
            order_data = self.place(symbol, side, qty or 1, "limit", limit_price)
            log_order("replace_fallback", {**order_data, "error": str(e)})
            return order_data

    # ------------------------------------------------------------------
    def get_status(self, order_id: str) -> Dict[str, Any]:
        if os.getenv("DRY_RUN", "").lower() == "true":
            order = _DRYRUN_ORDERS.get(order_id)
            if not order:
                raise ValueError("order_id is missing or unknown in DRY_RUN mode")
            time.sleep(random.uniform(0.1, 0.4))
            snapshot = dict(order)
            log_order("status_dryrun", snapshot)
            return snapshot

        o = self._try(self.api.get_order, order_id)
        order_data = self._normalize_order(o)
        log_order("status", order_data)
        return order_data
