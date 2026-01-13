"""
ai/utils/alpaca_client.py

Unified Alpaca client adapter for AITradeBot (Phase 92+)

- Supports PAPER + LIVE trading
- Provides unified market data API:
      get_last_price(symbol)
      get_quote(symbol)
      get_snapshot(symbol)
- Provides consistent order submission wrappers
- Backwards compatible with legacy class name "AlpacaClient"
"""

from __future__ import annotations

import logging
import os
import pandas as pd
from typing import Any, Dict, Optional, List

from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

log = logging.getLogger("AlpacaClientAdapter")

def adapter_place_order(*args, **kwargs):
    return AlpacaClientAdapter.place_order(*args, **kwargs)


class AlpacaClientAdapter:
    """
    Phase-92 unified Alpaca client for execution and market data.

    Accepts *args/**kwargs so older runners passing config objects
    do NOT break.
    """

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def __init__(self, *args, **kwargs) -> None:
        api_key = (
            os.getenv("ALPACA_API_KEY") or
            os.getenv("APCA_API_KEY_ID")
        )
        api_secret = (
            os.getenv("ALPACA_API_SECRET") or
            os.getenv("APCA_API_SECRET_KEY")
        )
        base_url = os.getenv(
            "ALPACA_API_BASE_URL",
            os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        )

        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Alpaca API credentials. "
                "Ensure ALPACA_API_KEY / ALPACA_API_SECRET or "
                "APCA_API_KEY_ID / APCA_API_SECRET_KEY exist."
            )

        try:
            self.api = REST(api_key, api_secret, base_url, api_version="v2")
            env_label = "paper" if "paper" in base_url else "live"
            log.info(f"🔗 AlpacaClientAdapter initialized ({env_label})")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Alpaca REST client: {e}")

    # ------------------------------------------------------------
    # OHLCV Loader for Phase 101 (Real Data RL Training)
    # ------------------------------------------------------------
    def get_ohlcv(self, symbol: str, start: str, end: str, timeframe: str = "1Min") -> pd.DataFrame:
        """
        Returns a pandas DataFrame with columns:
            ['open', 'high', 'low', 'close', 'volume']

        Uses Alpaca /v2/stocks/bars API via the free IEX feed.
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment="raw",
                feed="iex",  # IMPORTANT: use free IEX feed, not SIP
            ).df
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OHLCV from Alpaca: {e}")

        # Extract symbol level if multi-index
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol)

        return bars[["open", "high", "low", "close", "volume"]].copy()

    # ---------------------------------------------------------
    # ACCOUNT
    # ---------------------------------------------------------
    def get_account(self) -> Any:
        try:
            return self.api.get_account()
        except Exception as e:
            log.error(f"Failed to fetch Alpaca account: {e}")
            return None

    # ---------------------------------------------------------
    # ORDER SUBMISSION (supports both dict and explicit args)
    # ---------------------------------------------------------
    def place_order(
        self,
        order_or_symbol: Any,
        qty: Optional[float] = None,
        side: Optional[str] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase-92 unified order handler.
        Accepts either:
            • place_order(order_dict)  (SmartOrderRouter)
            • place_order(symbol, qty, side, ...)

        ALWAYS returns a standardized dictionary.
        """

        symbol = None

        # -----------------------------------------------------
        # CASE 1 — Called as place_order(order_dict)
        # -----------------------------------------------------
        if isinstance(order_or_symbol, dict) and qty is None:
            o = order_or_symbol  # rename for clarity
            symbol = o.get("symbol")
            qty = o.get("qty")
            side = o.get("side")
            order_type = o.get("order_type", "market")
            limit_price = o.get("limit_price")
            time_in_force = o.get("time_in_force", "day")
            client_order_id = o.get("client_order_id")

        # -----------------------------------------------------
        # CASE 2 — Called as place_order(symbol, qty, side, ...)
        # -----------------------------------------------------
        else:
            symbol = order_or_symbol

        # Validate minimal fields
        if not symbol or qty is None or not side:
            return {
                "ok": False,
                "id": "",
                "raw": None,
                "error": f"Missing required order fields (symbol={symbol}, qty={qty}, side={side})",
            }

        # Build Alpaca params
        params = {
            "symbol": symbol,
            "qty": qty,
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force,
        }
        if order_type.lower() == "limit":
            params["limit_price"] = limit_price
        if client_order_id:
            params["client_order_id"] = client_order_id

        try:
            log.info(
                f"📤 Alpaca place_order: {side.upper()} {symbol} x {qty} "
                f"type={order_type.upper()} limit={limit_price}"
            )
            resp = self.api.submit_order(**params)

            return {
                "ok": True,
                "id": resp.id,
                "raw": resp._raw or resp,
                "error": None,
            }

        except APIError as e:
            msg = f"APIError: {getattr(e, 'message', str(e))}"
            log.error(f"Order failed: {msg}")
            return {"ok": False, "id": "", "raw": getattr(e, "error", None), "error": msg}

        except Exception as e:
            msg = f"unexpected error placing order: {e}"
            log.error(f"Order failed: {msg}")
            return {"ok": False, "id": "", "raw": None, "error": msg}

    # ---------------------------------------------------------
    # ORDER CANCEL / MULTI
    # ---------------------------------------------------------
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            log.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        try:
            self.api.cancel_all_orders()
            return True
        except Exception as e:
            log.error(f"Failed to cancel all orders: {e}")
            return False

    # ---------------------------------------------------------
    # MARKET DATA (Unified Phase-92)
    # ---------------------------------------------------------
    def get_last_price(self, symbol: str) -> float:
        """
        Latest trade price (real-time). Used by Phase 26 runner.
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            log.error(f"Failed to fetch last price for {symbol}: {e}")
            return 0.0

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Best bid/ask + mid-price.
        """
        try:
            q = self.api.get_latest_quote(symbol)
            bid = float(q.bidprice)
            ask = float(q.askprice)
            mid = (bid + ask) / 2 if (bid and ask) else 0.0
            return {"bid": bid, "ask": ask, "mid": mid}
        except Exception as e:
            log.error(f"Failed to fetch quote for {symbol}: {e}")
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0}

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Snapshot: OHLC + last trade + quote.
        Used by advanced execution models (Phase 92.3/93).
        """
        try:
            snap = self.api.get_snapshot(symbol)
            return {
                "last": float(snap.latest_trade.p) if snap.latest_trade else 0.0,
                "bid": float(snap.latest_quote.bp) if snap.latest_quote else 0.0,
                "ask": float(snap.latest_quote.ap) if snap.latest_quote else 0.0,
                "volume": snap.daily_bar.v if snap.daily_bar else 0.0,
                "open": snap.daily_bar.o if snap.daily_bar else 0.0,
                "high": snap.daily_bar.h if snap.daily_bar else 0.0,
                "low": snap.daily_bar.l if snap.daily_bar else 0.0,
                "close": snap.daily_bar.c if snap.daily_bar else 0.0,
            }
        except Exception as e:
            log.error(f"Failed to fetch snapshot for {symbol}: {e}")
            return {}

    # ---------------------------------------------------------
    # POSITIONS
    # ---------------------------------------------------------
    def list_positions(self) -> List[Any]:
        try:
            return self.api.list_positions()
        except Exception as e:
            log.error(f"Failed to list positions: {e}")
            return []

    def get_positions(self) -> List[Any]:
        """Backwards-compatible alias (Phase 10–26 used this)."""
        return self.list_positions()

    def close_position(self, symbol: str) -> bool:
        try:
            self.api.close_position(symbol)
            return True
        except Exception as e:
            log.error(f"Failed to close position for {symbol}: {e}")
            return False


# ----------------------------------------------------------------------
# Backwards compatibility for older imports
# ----------------------------------------------------------------------
AlpacaClient = AlpacaClientAdapter
