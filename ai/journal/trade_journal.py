import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("data/reports/trade_journal.csv")
JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = [
    "ts", "symbol", "side", "qty", "fill_price", "order_type",
    "broker", "route_rank", "route_score", "slippage", "spread",
    "latency_ms", "pnl", "action_source", "policy_version",
    "entry_reason", "exit_reason", "notes"
]


class TradeJournal:
    """Human-readable trade journal."""

    @staticmethod
    def append(row: Dict[str, Any]) -> None:
        """Write row to CSV journal."""
        row["ts"] = datetime.utcnow().isoformat()

        try:
            file_exists = JOURNAL_PATH.exists()

            with JOURNAL_PATH.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Trade journal write failed: {e}")
