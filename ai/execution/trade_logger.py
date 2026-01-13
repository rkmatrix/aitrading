import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from ai.utils.log_utils import get_logger

LOG = get_logger("TradeLogger")

class TradeLogger:
    """
    Append every executed or rejected trade into a CSV file
    for downstream PnL or analytics modules.
    """

    def __init__(self, out_dir: str = "data/trades"):
        self.path = Path(out_dir) / "trade_log.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc",
                    "symbol",
                    "side",
                    "qty",
                    "venue",
                    "status",
                    "avg_price",
                    "reason",
                    "strength",
                    "order_id",
                ])
            LOG.info("Created trade log file at %s", self.path)

    def log(self, trade: dict):
        """Append one trade record."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            trade.get("symbol"),
            trade.get("side"),
            trade.get("qty"),
            trade.get("venue"),
            trade.get("status"),
            trade.get("avg_price"),
            trade.get("reason"),
            trade.get("strength"),
            trade.get("order_id"),
        ]
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            LOG.info("Logged trade %s %s %s@%s",
                     trade.get("side"),
                     trade.get("qty"),
                     trade.get("symbol"),
                     trade.get("venue"))
        except Exception as e:
            LOG.error("Failed to write trade log: %s", e)