"""
ai/execution/broker_alpaca_live.py
----------------------------------
REAL Alpaca Paper Trading Broker for SmartOrderRouter v4 and Phase 26.

Fixes:
- Missing import for tradeapi
- Missing AlpacaClient class (runner requires this name)
- Prevents fallback to STUB broker
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from tools.env_loader import ensure_env_loaded, validate_api_keys
from ai.execution.retry_handler import RetryableBrokerCall, RetryConfig, CircuitBreakerConfig

# ✅ Alpaca SDK import
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)


# =====================================================================
#  REQUIRED BY Phase-26 RUNNER
# =====================================================================
class AlpacaClient:
    """
    INTERNAL WRAPPER used by Phase-26 and SmartOrderRouter.

    Provides:
        - get_last_price(symbol)
        - get_account()
        - get_positions()
        - get_open_position_map()
        - place_order()

    Delegates low-level order sending to AlpacaLiveBroker.
    """

    def __init__(self):
        ensure_env_loaded(".env")

        self.mode = os.getenv("MODE", "PAPER").upper()

        # Validate API keys (fail fast if missing)
        validation_result = validate_api_keys(mode=self.mode, fail_fast=True)
        base_url = validation_result["base_url"]
        key_id = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        logger.info("🔌 Initializing AlpacaClient (%s)" % self.mode)

        # Instance logger + default timeframe
        self.log = logger
        self._timeframe = "1Min"

        # REAL Alpaca REST client
        self.client = tradeapi.REST(
            key_id=key_id,
            secret_key=secret_key,
            base_url=base_url,
        )

        # high-level wrapper (orders, fills, status)
        self.live = AlpacaLiveBroker(self.client, self.mode)
        
        # Retry handler with circuit breaker
        retry_config = RetryConfig(max_retries=3, initial_delay=1.0, max_delay=60.0)
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
        )
        self.retry_handler = RetryableBrokerCall(
            retry_config=retry_config,
            circuit_config=circuit_config,
        )

    # --------------------------------------------------------------
    # PRICE FEED
    # --------------------------------------------------------------
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Return the last traded price for `symbol`, or None if unavailable.
        Handles empty bar responses gracefully instead of raising IndexError.
        """
        api = self.client
        timeframe = getattr(self, "_timeframe", "1Min")

        try:
            bars = api.get_bars(
                symbol,
                timeframe,
                limit=1,
            )
        except Exception as e:
            # Log and return None so the caller can skip this symbol for the tick
            self.log.error("❌ get_last_price API error for %s: %s", symbol, e)
            return None

        # Alpaca SDK sometimes returns [] when market is closed / no data
        if not bars:
            self.log.warning("⚠️ get_last_price: no bars returned for %s; skipping.", symbol)
            return None

        try:
            bar = bars[0]
            # Adjust .c / .close depending on your Alpaca object
            price = float(getattr(bar, "c", getattr(bar, "close", None)))
        except Exception as e:
            self.log.error("❌ get_last_price: failed to parse bar for %s: %s", symbol, e)
            return None

        return price

    # --------------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------------
    def get_account(self) -> Dict[str, Any]:
        """Get account information with response validation and retry logic."""
        def _get_account_internal():
            acct = self.client.get_account()
            if acct is None:
                raise ValueError("Account response is None")
            
            # Validate response structure
            equity = getattr(acct, "equity", None)
            buying_power = getattr(acct, "buying_power", None)
            
            if equity is None or buying_power is None:
                raise ValueError(f"Invalid account response: missing equity or buying_power")
            
            equity_float = float(equity)
            buying_power_float = float(buying_power)
            
            # Sanity checks
            if equity_float < 0:
                raise ValueError(f"Invalid equity value: {equity_float}")
            if buying_power_float < 0:
                self.log.warning("⚠️ Negative buying power: %s", buying_power_float)
            
            return {
                "equity": equity_float,
                "buying_power": buying_power_float,
            }
        
        # Use retry handler
        success, result, error = self.retry_handler.execute(_get_account_internal)
        
        if not success:
            self.log.error("❌ Failed to get account after retries: %s", error, exc_info=True)
            raise RuntimeError(f"Failed to get account: {error}")
        
        return result

    # --------------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------------
    def get_positions(self):
        """Get positions with response validation."""
        try:
            positions = self.client.list_positions()
            if positions is None:
                return []
            
            results = []
            for p in positions:
                try:
                    # Validate position structure
                    symbol = getattr(p, "symbol", None)
                    qty = getattr(p, "qty", None)
                    price = getattr(p, "avg_entry_price", None)
                    
                    if symbol is None:
                        self.log.warning("⚠️ Position missing symbol, skipping")
                        continue
                    
                    if qty is None:
                        self.log.warning("⚠️ Position %s missing qty, skipping", symbol)
                        continue
                    
                    results.append(
                        {
                            "symbol": str(symbol),
                            "qty": float(qty),
                            "price": float(price) if price is not None else 0.0,
                        }
                    )
                except Exception as e:
                    self.log.warning("⚠️ Failed to parse position: %s", e)
                    continue
            
            return results
        except Exception as e:
            self.log.error("❌ Failed to get positions: %s", e, exc_info=True)
            return []

    def get_open_position_map(self) -> Dict[str, Any]:
        out = {}
        for p in self.client.list_positions():
            out[p.symbol] = {
                "qty": float(p.qty),
                "price": float(p.avg_entry_price),
            }
        return out

    # --------------------------------------------------------------
    # ORDER ROUTING (delegated to AlpacaLiveBroker)
    # --------------------------------------------------------------
    def place_order(self, order: Dict[str, Any]):
        return self.live.submit_order(order)


# =====================================================================
#  Lower-level broker (your existing logic)
# =====================================================================
class AlpacaLiveBroker:
    """
    Clean wrapper around Alpaca REST API.
    Responsible for:
        - submit_order
        - get_order_status
        - extract_fill
    """

    def __init__(self, client: tradeapi.REST, mode: str = "PAPER"):
        self.client = client
        self.mode = mode

    @classmethod
    def from_env(cls) -> "AlpacaLiveBroker":
        ensure_env_loaded(".env")

        mode = os.getenv("MODE", "PAPER").upper()
        key_id = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

        if not key_id or not secret_key:
            raise RuntimeError("Missing Alpaca API keys.")

        client = tradeapi.REST(
            key_id=key_id,
            secret_key=secret_key,
            base_url=base_url,
        )

        logger.info("🔌 AlpacaLiveBroker initialized (mode=%s)", mode)
        return cls(client, mode)

    # --------------------------------------------------------------
    # ORDERS
    # --------------------------------------------------------------
    def submit_order(self, order: Dict[str, Any]) -> Any:
        symbol = order["symbol"]
        side = order["side"].lower()
        qty = float(order["qty"])
        order_type = order.get("order_type", "market").lower()
        limit_price = order.get("limit_price")
        client_order_id = order.get("client_order_id")

        try:
            if order_type == "market":
                resp = self.client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc",
                    client_order_id=client_order_id,
                )

            elif order_type == "limit":
                resp = self.client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="limit",
                    limit_price=float(limit_price),
                    time_in_force="gtc",
                    client_order_id=client_order_id,
                )

            else:
                raise ValueError(f"Unsupported order_type={order_type}")

            # Validate response structure
            if resp is None:
                raise ValueError("Order response is None")
            
            order_id = getattr(resp, "id", None)
            if order_id is None:
                raise ValueError("Order response missing id")
            
            # Validate order status
            status = getattr(resp, "status", None)
            if status is None:
                logger.warning("⚠️ Order response missing status field")
            
            logger.info("📨 Alpaca accepted order → id=%s, status=%s", order_id, status)
            return resp

        except Exception as e:
            logger.error("❌ Alpaca order error: %s", e, exc_info=True)
            return {"error": str(e), "order_submitted": False, "status": "ERROR"}

    # --------------------------------------------------------------
    def get_order_status(self, order_id: str):
        try:
            o = self.client.get_order(order_id)
            return {
                "id": o.id,
                "symbol": o.symbol,
                "status": o.status,
                "filled_qty": o.filled_qty,
                "filled_avg_price": o.filled_avg_price,
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at),
            }
        except Exception as e:
            logger.warning("⚠️ Failed to fetch status %s: %s", order_id, e)
            return None

    # --------------------------------------------------------------
    def extract_fill(self, resp: Any) -> Dict[str, Any]:
        if resp is None:
            return {}

        if isinstance(resp, dict) and "error" in resp:
            return {}

        try:
            return {
                "price": float(getattr(resp, "filled_avg_price", None)),
                "qty": float(getattr(resp, "filled_qty", None)),
            }
        except Exception:
            return {}
