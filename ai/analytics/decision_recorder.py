# ai/analytics/decision_recorder.py
"""
Phase 63 – Decision Recorder & Explainability Log

Captures:
    • AgentContext snapshot
    • FusedDecision from MultiAgentBrain
    • ExecutionResult (from ExecutionPipeline)
and appends them to:
    • JSONL file (rich detail)
    • optional CSV file (tabular summary)

Intended for:
    • Offline analysis
    • Dashboards
    • Replay / retraining later
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ai.agents.base_agent import AgentContext
from ai.agents.votes import FusedDecision
from ai.execution.execution_pipeline import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """
    A single decision snapshot.

    Fields:
        ts:           Timestamp when record was written (UTC)
        mode:         DEMO | PAPER | LIVE (or similar)
        symbol:       Instrument symbol
        context:      Minimal context snapshot (price, equity, position qty, etc.)
        decision:     FusedDecision.to_dict() output
        order_sent:   Whether an order was sent
        order_meta:   Broker/router result or error
        extra:        Optional additional info
    """

    ts: datetime
    mode: str
    symbol: str
    context: Dict[str, Any]
    decision: Dict[str, Any]
    order_sent: bool
    order_meta: Optional[Dict[str, Any]]
    extra: Dict[str, Any]


class DecisionRecorder:
    """
    Handles appending DecisionRecord entries to JSONL (and optional CSV).

    JSONL:
        • One JSON object per line (rich, nested structure)
    CSV:
        • Flattened summary (symbol, action, conf, conflict, order_sent, qty, etc.)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        csv_path: str | Path | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path) if csv_path else None

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV header will be written on first write if file does not exist
        self._csv_initialized = False

    # ---- Public API -------------------------------------------------------

    def record(
        self,
        ctx: AgentContext,
        decision: FusedDecision,
        result: ExecutionResult,
        *,
        mode: str = "DEMO",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Build a DecisionRecord and persist it.
        """
        ts = datetime.utcnow()

        ctx_snapshot = self._context_to_dict(ctx)
        decision_dict = decision.to_dict()
        order_meta = result.order_meta if result.order_meta is not None else None

        record = DecisionRecord(
            ts=ts,
            mode=str(mode),
            symbol=ctx.symbol,
            context=ctx_snapshot,
            decision=decision_dict,
            order_sent=result.order_sent,
            order_meta=order_meta,
            extra=extra or {},
        )

        self._append_jsonl(record)
        self._append_csv(record)

    # ---- Internal helpers -------------------------------------------------

    def _context_to_dict(self, ctx: AgentContext) -> Dict[str, Any]:
        return {
            "ts": ctx.ts.isoformat(),
            "symbol": ctx.symbol,
            "price": ctx.price,
            "position": dict(ctx.position or {}),
            "portfolio": dict(ctx.portfolio or {}),
            "extra": dict(ctx.extra or {}),
        }

    def _append_jsonl(self, record: DecisionRecord) -> None:
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                obj = {
                    "ts": record.ts.isoformat(),
                    "mode": record.mode,
                    "symbol": record.symbol,
                    "context": record.context,
                    "decision": record.decision,
                    "order_sent": record.order_sent,
                    "order_meta": record.order_meta,
                    "extra": record.extra,
                }
                f.write(json.dumps(obj) + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("💥 Failed to append decision to JSONL (%s): %s", self.jsonl_path, exc, exc_info=True)

    def _append_csv(self, record: DecisionRecord) -> None:
        if not self.csv_path:
            return

        summary = self._build_csv_summary(record)

        try:
            file_exists = self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(summary)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("💥 Failed to append decision to CSV (%s): %s", self.csv_path, exc, exc_info=True)

    def _build_csv_summary(self, record: DecisionRecord) -> Dict[str, Any]:
        d = record.decision
        votes = d.get("votes", [])
        num_votes = len(votes)

        return {
            "ts": record.ts.isoformat(),
            "mode": record.mode,
            "symbol": record.symbol,
            "final_action": d.get("final_action"),
            "final_size": d.get("final_size"),
            "final_broker": d.get("final_broker"),
            "fused_conf": d.get("fused_conf"),
            "conflict_score": d.get("conflict_score"),
            "num_votes": num_votes,
            "order_sent": record.order_sent,
            "order_status": (record.order_meta or {}).get("status"),
        }


__all__ = ["DecisionRecord", "DecisionRecorder"]
