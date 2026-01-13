"""
Alpaca Broker Adapter (Paper or Live)
-------------------------------------
This adapter executes orders via Alpaca’s official Trading API.

✅ Automatically loads credentials from your `.env` file if not found in environment.
✅ Works in both PAPER and LIVE modes.
✅ Handles async order submission + status polling.

Env vars (can be in `.env`):
    ALPACA_API_KEY
    ALPACA_API_SECRET
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

import os, asyncio, logging, time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ------------------------------------------------------------
# Auto-load .env file if present
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
    logging.info("🌱 Loaded environment variables from .env")
except Exception as e:
    logging.warning(f"⚠️ Could not auto-load .env file: {e}")

# ------------------------------------------------------------
# Load credentials
# ------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
IS_PAPER = "paper" in BASE_URL

if not (API_KEY and API_SECRET):
    logging.warning("⚠️ Alpaca credentials missing – broker_alpaca will not trade.")

# Initialize Alpaca trading client
try:
    trading_client = TradingClient(API_KEY, API_SECRET, paper=IS_PAPER)
except Exception as e:
    trading_client = None
    logging.error(f"💥 Alpaca TradingClient init failed: {e}")

# ------------------------------------------------------------
async def execute(order: dict) -> dict:
    """
    Place a market order and wait for fill confirmation.
    Returns a normalized result dict.
    """
    symbol = order.get("symbol")
    qty = int(order.get("qty", 1))
    side = OrderSide.BUY if order.get("side", "").upper() == "BUY" else OrderSide.SELL
    tif = TimeInForce.DAY

    if not trading_client:
        logging.warning("⚠️ Alpaca client not initialized – using dry-run fallback.")
        await asyncio.sleep(0.05)
        return {
            "symbol": symbol,
            "order_id": f"DRY_{int(time.time()*1000)}",
            "filled_qty": float(qty),
            "side": side.value.upper(),
            "pnl": 0.0,
            "slippage": 0.0,
            "status": "filled",
            "broker": "alpaca_dry",
        }

    try:
        logging.info(f"📈 Submitting {side.value.upper()} {qty}@{symbol} via Alpaca ({'PAPER' if IS_PAPER else 'LIVE'})")
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=tif)
        resp = trading_client.submit_order(req)

        order_id = resp.id
        status = resp.status
        filled_qty = 0
        start_time = time.time()

        # Poll until filled or timeout (20s)
        while status not in ("filled", "canceled", "rejected"):
            await asyncio.sleep(1.5)
            info = trading_client.get_order_by_id(order_id)
            status = info.status
            filled_qty = float(info.filled_qty or 0)
            logging.info(f"🔄 {symbol} order {order_id} status={status} filled={filled_qty}")
            if time.time() - start_time > 20:
                logging.warning(f"⏰ Timeout waiting for {symbol} fill (status={status})")
                break

        logging.info(f"✅ {symbol} order {order_id} final status={status}")

        return {
            "symbol": symbol,
            "order_id": order_id,
            "filled_qty": filled_qty,
            "side": side.value.upper(),
            "pnl": 0.0,
            "slippage": 0.0,
            "status": status,
            "broker": "alpaca",
        }

    except Exception as e:
        logging.error(f"💥 Alpaca execute() failed: {e}")
        return {
            "symbol": symbol,
            "filled_qty": 0.0,
            "side": side.value.upper(),
            "error": str(e),
            "broker": "alpaca",
        }

# Router interface expected by SmartOrderRouter
broker = {"execute": execute, "name": "alpaca", "mode": "PAPER" if IS_PAPER else "LIVE"}

# ------------------------------------------------------------
if __name__ == "__main__":
    # Quick manual test
    async def _test():
        res = await execute({"symbol": "AAPL", "side": "BUY", "qty": 1, "order_type": "MARKET"})
        print(res)
    asyncio.run(_test())
