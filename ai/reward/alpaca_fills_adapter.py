# ai/reward/alpaca_fills_adapter.py
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from .sources import Event, EventSource

log = logging.getLogger("AlpacaFillsSource")


class AlpacaFillsSource(EventSource):
    """
    EventSource implementation backed by Alpaca's Trading API.

    It polls *closed* orders via alpaca-py and converts any *new* fully/partially
    filled orders into reward Events.

    NOTE:
        - This is a *scaffold*. PnL, slippage, and risk are conservative defaults
          (0.0) for now. You can wire in real PnL from your account or portfolio
          history later.
        - By default, if Alpaca is not configured, poll() simply returns [] and
          logs a warning, so the loop will not crash.

    Env / config:
        - ALPACA_API_KEY or APCA_API_KEY_ID
        - ALPACA_SECRET_KEY or APCA_API_SECRET_KEY
        - cfg['source']['alpaca']:
            paper: bool (default True)
            max_orders_per_poll: int (default 50)
    """

    def __init__(self, symbols: List[str], cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.symbols = list(symbols or [])
        self._seen_order_ids: set[str] = set()

        # Try importing alpaca-py lazily so the rest of the system works even
        # without it installed.
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
        except Exception as e:  # pragma: no cover - import guard
            log.error("alpaca-py not installed or failed to import: %s", e)
            self._TradingClient = None
            self._GetOrdersRequest = None
            self._QueryOrderStatus = None
            self._client = None
            return

        self._TradingClient = TradingClient
        self._GetOrdersRequest = GetOrdersRequest
        self._QueryOrderStatus = QueryOrderStatus

        # API keys: prefer explicit config, then env vars.
        api_key = (
            cfg.get("api_key")
            or os.getenv("ALPACA_API_KEY")
            or os.getenv("APCA_API_KEY_ID")
        )
        secret_key = (
            cfg.get("secret_key")
            or os.getenv("ALPACA_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
        )
        paper = bool(cfg.get("paper", True))

        if not api_key or not secret_key:
            log.warning(
                "AlpacaFillsSource: No API keys configured; source will be inactive. "
                "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or APCA_API_KEY_ID / APCA_API_SECRET_KEY."
            )
            self._client = None
        else:
            self._client = self._TradingClient(
                api_key, secret_key, paper=paper, raw_data=False
            )

        self._max_orders_per_poll = int(cfg.get("max_orders_per_poll", 50))

    # ------------------------------------------------------------------ utils

    def _is_fill_status(self, status: str) -> bool:
        s = (status or "").lower()
        return s in ("filled", "partially_filled", "done_for_day")

    # ----------------------------------------------------------------- polling

    def poll(self, max_events: int) -> List[Event]:
        """
        Fetch recent closed orders from Alpaca and convert new ones to Events.

        For now:
            - px: filled_avg_price (or 0.0)
            - position: signed filled_qty (+ for buy, - for sell)
            - realized_pnl, unrealized_pnl, slippage, risk: 0.0 (placeholders)
        """
        if not self._client or not self._GetOrdersRequest:
            # Not configured or alpaca-py missing
            return []

        limit = min(max_events, self._max_orders_per_poll)
        if limit <= 0:
            return []

        try:
            req = self._GetOrdersRequest(
                status=self._QueryOrderStatus.CLOSED,
                limit=limit,
                nested=False,
                symbols=self.symbols or None,
            )
            orders = self._client.get_orders(req)
        except Exception as e:  # pragma: no cover - network
            log.error("AlpacaFillsSource: get_orders failed: %s", e)
            return []

        events: List[Event] = []
        for o in orders:
            oid = getattr(o, "id", None)
            if not oid or oid in self._seen_order_ids:
                continue

            self._seen_order_ids.add(oid)
            status = str(getattr(o, "status", "")).lower()
            if not self._is_fill_status(status):
                continue

            symbol = getattr(o, "symbol", None)
            if self.symbols and symbol not in self.symbols:
                continue

            side = str(getattr(o, "side", "buy")).lower()
            filled_qty = float(
                getattr(o, "filled_qty", getattr(o, "qty", 0.0)) or 0.0
            )
            if side == "sell":
                filled_qty = -filled_qty

            px = float(
                getattr(
                    o, "filled_avg_price", getattr(o, "limit_price", 0.0)
                )
                or 0.0
            )
            ts_obj = getattr(o, "filled_at", None) or getattr(o, "submitted_at", None)
            ts_unix = ts_obj.timestamp() if ts_obj is not None else time.time()

            # Scaffold: fill these with neutral values for now
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            slippage = 0.0
            risk = 0.0

            events.append(
                Event(
                    ts=ts_unix,
                    symbol=symbol,
                    px=px,
                    position=filled_qty,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    slippage=slippage,
                    risk=risk,
                    meta={
                        "order_id": oid,
                        "side": side,
                        "raw_status": status,
                    },
                )
            )

        return events

    def close(self) -> None:
        # TradingClient does not need explicit close; included for symmetry.
        pass
