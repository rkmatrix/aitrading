# ai/execution/router_bridge.py
from __future__ import annotations
import logging, inspect, time, csv
from pathlib import Path
from typing import Dict, Any
from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class RouteResult:
    def __init__(self, ok: bool, detail: str, order: Dict[str, Any], latency: float = 0.0):
        self.ok = ok
        self.detail = detail
        self.order = order
        self.latency = latency


class RouterBridge:
    """
    SmartOrderRouter wrapper + telemetry + failover.
    If Alpaca 401/timeout occurs, automatically switches to dummy mode and
    alerts via Telegram.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self._router = None
        self._dummy_mode = False
        self._metrics_csv = Path("data/reports/router_metrics.csv")
        self._metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics_csv.exists():
            with open(self._metrics_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp","symbol","side","qty","status","latency_sec","detail"]
                )
        self._init_router()

    # ------------------------------------------------------------------
    def _init_router(self):
        try:
            from ai.execution.smart_order_router import SmartOrderRouter
            self._router = SmartOrderRouter(self.cfg)
            logger.info("✅ RouterBridge: SmartOrderRouter loaded.")
        except Exception as e:
            self._router = None
            self._dummy_mode = True
            logger.warning("⚠️ Router init failed (%s). Using dummy router.", e)
            notify(f"⚠️ Router init failed: {e}\nUsing dummy mode.", kind="system")

    # ------------------------------------------------------------------
    def route(self, order: Dict[str, Any]) -> RouteResult:
        start = time.perf_counter()
        ok = True
        detail = ""
        latency = 0.0

        if self._dummy_mode or self._router is None:
            logger.info("[DUMMY_ROUTE] %s", order)
            detail = "dummy_mode"
            self._write_metrics(order, True, 0.0, detail)
            return RouteResult(True, detail, order, 0.0)

        try:
            sig = inspect.signature(self._router.route)
            params = list(sig.parameters.keys())

            if len(params) == 2 and "order" in params:
                self._router.route(order)
                detail = "sent_dict"
            elif len(params) == 4 and {"symbol","side","qty"}.issubset(params):
                self._router.route(order["symbol"], order["side"], order["qty"])
                detail = "sent_args3"
            else:
                self._router.route(
                    order["symbol"], order["side"], order["qty"],
                    order.get("order_type","MARKET"),
                    order.get("limit_price"),
                    order.get("tag","phase53_blend")
                )
                detail = "sent_args6"
            latency = time.perf_counter() - start
            logger.info("🟢 Routed %s (%.4fs) %s", detail, latency, order)

        except Exception as e:
            latency = time.perf_counter() - start
            detail = str(e)
            ok = False
            logger.error("Router error after %.3fs: %s", latency, e)
            # --- auto-fallback for 401 or timeout ---
            if "401" in detail or "unauthorized" in detail.lower() or "timeout" in detail.lower():
                self._dummy_mode = True
                notify(f"⚠️ Router error: {detail}\nSwitching to dummy mode.", kind="system")
                logger.warning("🔁 RouterBridge switched to dummy mode after failure.")
        self._write_metrics(order, ok, latency, detail)
        return RouteResult(ok, detail, order, latency)

    # ------------------------------------------------------------------
    def _write_metrics(self, order: Dict[str, Any], ok: bool, latency: float, detail: str):
        try:
            with open(self._metrics_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    order.get("symbol"),
                    order.get("side"),
                    order.get("qty"),
                    "OK" if ok else "ERR",
                    round(latency,4),
                    detail,
                ])
        except Exception as e:
            logger.debug("Metrics CSV write failed: %s", e)
