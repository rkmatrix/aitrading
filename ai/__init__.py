# ai/__init__.py
"""
AITradeBot package initializer
- Automatically loads .env on any import of the ai.* namespace.
- Verifies env (PAPER/LIVE) and logs a one-time startup line.
- Sends a one-time Telegram startup alert with Alpaca equity & buying power.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Telegram notifier (safe to import here; it won't spam unless called)
try:
    from tools.telegram_alerts import notify
except Exception:
    # If tools are not available yet during early setup, we fall back gracefully.
    def notify(*args, **kwargs):
        pass

logger = logging.getLogger("AITradeBotEnv")


def _auto_env_load():
    env_path = Path(__file__).resolve().parent.parent / ".env"

    # Load .env if not already loaded
    if not os.getenv("APCA_API_KEY_ID"):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)

    mode = (os.getenv("MODE") or "PAPER").upper()
    base = (os.getenv("APCA_API_BASE_URL") or "").lower()
    key = os.getenv("APCA_API_KEY_ID") or ""
    env_ok = bool(key and base)

    # Setup logging (only once)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if env_ok:
        if "paper" in base:
            env_type = "PAPER TRADING"
        elif "live" in base:
            env_type = "LIVE TRADING"
        else:
            env_type = "CUSTOM ENDPOINT"

        # Base startup line
        logger.info(f"✅ .env loaded → Mode={mode} | Env={env_type} | Key={key[:6]}***")

        # ---- One-time Alpaca equity self-check + Telegram startup alert ----
        eq_line = ""
        try:
            import requests
            r = requests.get(
                f"{os.getenv('APCA_API_BASE_URL')}/v2/account",
                headers={
                    "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
                    "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
                },
                timeout=6,
            )
            if r.status_code == 200:
                acc = r.json()
                eq = float(acc.get("equity", 0))
                bp = float(acc.get("buying_power", 0))
                eq_line = f" | Equity=${eq:,.2f} | BP=${bp:,.2f}"
                logger.info(f"📈 Alpaca account equity = ${eq:,.2f} | Buying Power = ${bp:,.2f}")
            else:
                logger.warning(f"⚠️ Alpaca account check failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.warning(f"⚠️ Alpaca self-check skipped ({e})")

        # Single Telegram startup alert (includes equity/buying power if available)
        try:
            notify(f"🚀 AITradeBot started in {env_type} mode ({mode}){eq_line}", kind="system")
        except Exception as e:
            logger.warning(f"⚠️ Telegram startup alert failed: {e}")

    else:
        logger.warning("⚠️ .env not fully loaded or missing Alpaca credentials.")
        logger.warning(f"Expected at: {env_path}")


_auto_env_load()
