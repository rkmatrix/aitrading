import os
import json
import requests

TG_RESERVED = set(["_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"])

def _escape_md(text: str) -> str:
    out = []
    for ch in text:
        if ch in TG_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)

def send_telegram(text: str, token: str = None, chat_id: str = None, parse_mode: str = "MarkdownV2"):
    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Telegram disabled or credentials missing"
    if parse_mode == "MarkdownV2":
        text = _escape_md(text)
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text}"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def alert_trade(event: str, payload: dict, enabled: bool = True):
    if not enabled:
        return
    msg = f"🔔 *{event}*\n```\n{json.dumps(payload, indent=2)}\n```"
    send_telegram(msg)
