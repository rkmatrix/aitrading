import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

LEDGER_PATH = Path("data/runtime/order_ledger.jsonl")
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)


class OrderLedger:
    """
    Phase 71 — Order Ledger
    Machine-readable structured log of order lifecycle.
    """

    @staticmethod
    def log(event: Dict[str, Any]) -> None:
        """Append event to JSONL ledger."""
        try:
            event["_ts"] = datetime.utcnow().isoformat()
            with open(LEDGER_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Ledger write failed: {e}")

    @staticmethod
    def order_submitted(order: Dict[str, Any], route_info: Dict[str, Any]):
        OrderLedger.log({
            "type": "submitted",
            "order": order,
            "route_info": route_info
        })

    @staticmethod
    def order_rerouted(order_id: str, route_info: Dict[str, Any]):
        OrderLedger.log({
            "type": "reroute",
            "order_id": order_id,
            "route_info": route_info
        })

    @staticmethod
    def order_fill(order_id: str, fill: Dict[str, Any]):
        OrderLedger.log({
            "type": "fill",
            "order_id": order_id,
            "fill": fill
        })

    @staticmethod
    def order_error(order_id: str, error: str, context: Optional[Dict[str, Any]] = None):
        OrderLedger.log({
            "type": "error",
            "order_id": order_id,
            "error": error,
            "ctx": context
        })

    @staticmethod
    def order_complete(order_id: str, summary: Dict[str, Any]):
        OrderLedger.log({
            "type": "complete",
            "order_id": order_id,
            "summary": summary
        })
