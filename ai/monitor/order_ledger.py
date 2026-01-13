# ai/monitor/order_ledger.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderLedgerConfig:
    log_dir: str = "data/logs/orders"
    enabled: bool = True
    source: str = "phase26"
    pretty_print: bool = False   # if True, also mirror some events in normal log
    include_context: bool = True # include risk context / extra info in each line


class OrderLedger:
    """
    Phase 113 – Robust Order & Event Ledger

    Responsibilities:
    -----------------
    - Append every order / fill / cancel / error to a canonical JSONL file:
        data/logs/orders/YYYYMMDD_orders.jsonl
    - Each line is a single JSON object with fields:
        - ts, ts_iso, event, symbol, side, qty, price, order_id, client_order_id, ...
        - source, tag, equity, gross_exposure, meta, router_response
    """

    def __init__(self, cfg: OrderLedgerConfig) -> None:
        self.cfg = cfg
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = None
        self._current_path: Optional[Path] = None
        self._fh = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_file(self) -> None:
        today = datetime.utcnow().strftime("%Y%m%d")
        if today != self._current_date or self._fh is None:
            # roll file
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._current_date = today
            self._current_path = self.log_dir / f"{today}_orders.jsonl"
            self._fh = self._current_path.open("a", encoding="utf-8")
            logger.info("OrderLedger: writing to %s", self._current_path)

    def _write_line(self, record: Dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        self._ensure_file()
        try:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        except Exception:
            logger.exception("OrderLedger: failed to write record.")

    # ------------------------------------------------------------------
    # Generic event logger
    # ------------------------------------------------------------------
    def log_event(
        self,
        event: str,
        *,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
        tag: Optional[str] = None,
        equity: Optional[float] = None,
        gross_exposure: Optional[float] = None,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        router_response: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Lowest-level event logging API.
        """
        if not self.cfg.enabled:
            return

        now = datetime.utcnow()
        rec: Dict[str, Any] = {
            "ts": now.timestamp(),
            "ts_iso": now.isoformat(),
            "event": event,                 # e.g. "order_submit", "order_result", "error"
            "source": self.cfg.source,      # e.g. "phase26"
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "tag": tag,
            "equity": equity,
            "gross_exposure": gross_exposure,
            "order_id": order_id,
            "client_order_id": client_order_id,
        }

        if router_response is not None:
            rec["router_response"] = router_response

        if meta:
            rec["meta"] = meta

        self._write_line(rec)

        if self.cfg.pretty_print:
            logger.info(
                "OrderLedger[%s] %s %s %s @ %s qty=%.4f eq=%.2f tag=%s",
                event,
                symbol,
                side,
                order_id or "",
                price,
                qty or 0.0,
                equity or 0.0,
                tag or "",
            )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def log_order_submit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float | None,
        tag: str | None = None,
        equity: float | None = None,
        gross_exposure: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            "order_submit",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            tag=tag,
            equity=equity,
            gross_exposure=gross_exposure,
            meta=meta,
        )

    def log_order_result(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float | None,
        tag: str | None,
        equity_after: float | None,
        gross_exposure_after: float | None,
        router_response: Dict[str, Any] | None,
        pnl: float | None = None,
        trade_cost: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        m = dict(meta or {})
        if pnl is not None:
            m["pnl"] = float(pnl)
        if trade_cost is not None:
            m["trade_cost"] = float(trade_cost)

        self.log_event(
            "order_result",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            tag=tag,
            equity=equity_after,
            gross_exposure=gross_exposure_after,
            router_response=router_response,
            meta=m,
        )

    def log_error(
        self,
        *,
        message: str,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        m = dict(meta or {})
        m["message"] = message
        self.log_event(
            "error",
            symbol=symbol,
            side=side,
            meta=m,
        )
