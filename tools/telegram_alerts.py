"""
tools/telegram_alerts.py
Unified Telegram notifier for AITradeBot (PHASE 81 — CLEAN ALERT SYSTEM)

This patched version:
  ✔ Sends ONLY actual order execution alerts
  ✔ Blocks ALL noisy internal messages:
        - risk blocks
        - equity <= 0
        - guardian warnings
        - RL decision noise
        - system errors / debug spam

Environment:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    TELEGRAM_ENABLED=true/false
    TELEGRAM_ALERT_TYPES="orders"  <-- ONLY this needed!
"""

from __future__ import annotations
import os
import json
import logging
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# User wants *ONLY* execution alerts
ALERT_TYPES = {"orders"}   # hard override; ignore .env list entirely


# ------------------------
# Noise Filters
# ------------------------
NOISY_PATTERNS = [
    "equity <=",
    "refusing all trades",
    "order blocked by risk",
    "risk",
    "guardian",
    "kill",
    "dramatic",
    "block",
    "refused",
    "rl decision",
    "internal",
]


def _is_noisy(msg: str) -> bool:
    if not msg:
        return True
    lower = msg.lower()
    return any(p in lower for p in NOISY_PATTERNS)


# ------------------------
# Core Notify Function
# ------------------------
def notify(msg: str, *, kind: str = "system", meta: dict | None = None):
    """
    General dispatcher.
    Blocks everything except:
        kind == "orders"
    """

    if not TELEGRAM_ENABLED:
        return

    if kind not in ALERT_TYPES:
        return

    # Block noisy messages even if incorrectly tagged
    if _is_noisy(msg):
        return

    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured; cannot send alert.")
        return

    text = msg
    if meta:
        try:
            text += "\n" + json.dumps(meta, indent=2)
        except Exception:
            pass

    _send_telegram_message(text)


# ------------------------
# Actual Telegram Send
# ------------------------
def _send_telegram_message(text: str):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode(
            {"chat_id": CHAT_ID, "text": text}
        ).encode()

        req = urllib.request.Request(url, data=payload)
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)


# ------------------------
# Legacy Shim Functions (Order-Only)
# ------------------------
def send_trade_alert(msg: str, meta: dict | None = None):
    notify(msg, kind="orders", meta=meta)


def send_pnl_alert(msg: str, meta: dict | None = None):
    # disabled by design
    pass


def send_system_alert(msg: str, meta: dict | None = None):
    # disabled by design
    pass
