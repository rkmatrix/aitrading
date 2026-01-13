from __future__ import annotations
import os, json, urllib.request
from typing import Optional

# Escapes for MarkdownV2 (fixes your earlier Telegram parse errors)
_TELEGRAM_RESERVED = r'_*[]()~`>#+-=|{}.!'

def escape_md_v2(text: str) -> str:
    out = []
    for ch in text:
        if ch in _TELEGRAM_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)

def send_telegram(msg: str, parse_mode: str = "MarkdownV2") -> Optional[str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    if parse_mode == "MarkdownV2":
        msg = escape_md_v2(msg)
    data = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return "ok"
