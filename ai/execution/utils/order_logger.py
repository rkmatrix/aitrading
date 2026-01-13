import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

LOG_DIR = Path("data/orders")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _current_log_file() -> Path:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return LOG_DIR / f"orders_{today}.csv"


def log_order(event: str, order: Dict[str, Any]):
    """Append a single order event to today's CSV log file."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    row = {
        "timestamp": ts,
        "event": event,
        "order_id": order.get("order_id"),
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "type": order.get("type"),
        "limit_price": order.get("limit_price"),
        "filled_qty": order.get("filled_qty"),
        "filled_avg_price": order.get("filled_avg_price"),
        "status": order.get("status"),
        "tif": order.get("time_in_force"),
        "error": order.get("error"),
    }

    log_file = _current_log_file()
    file_exists = log_file.exists()
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
