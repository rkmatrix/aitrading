"""
ai/analytics/execution_alpha.py
--------------------------------

Phase 72 — Execution Alpha Analyzer

Reads:
    - data/reports/trade_journal.csv
    - data/runtime/order_ledger.jsonl

Computes:
    - Per-symbol execution stats (slippage, latency, spread)
    - Per-broker quality (avg slippage, error rate)
    - Route score calibration (does higher score actually mean better fills?)
    - Basic trade outcome distribution if PnL is present

Outputs:
    - Summary dict (for programmatic use)
    - Optional JSON/CSV reports for dashboards & later phases
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from statistics import mean

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("data/reports/trade_journal.csv")
LEDGER_PATH = Path("data/runtime/order_ledger.jsonl")
SUMMARY_JSON = Path("data/reports/phase72_execution_alpha_summary.json")


def _safe_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


class ExecutionAlphaAnalyzer:
    """
    Core analytics engine for execution quality.

    Typical use:
        analyzer = ExecutionAlphaAnalyzer()
        summary = analyzer.run()
    """

    def __init__(
        self,
        journal_path: Path = JOURNAL_PATH,
        ledger_path: Path = LEDGER_PATH,
    ) -> None:
        self.journal_path = journal_path
        self.ledger_path = ledger_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run full analysis pipeline.

        Returns:
            summary dict with keys:
                symbols, brokers, overall, route_score
        """
        trades = self._load_journal_rows()
        ledger_events = self._load_ledger_events()

        if not trades:
            logger.warning("Phase72: No trades found in trade_journal.csv")
        if not ledger_events:
            logger.info("Phase72: No ledger events found in order_ledger.jsonl")

        per_symbol = self._analyze_per_symbol(trades)
        per_broker = self._analyze_per_broker(trades)
        overall = self._analyze_overall(trades)
        route_score = self._analyze_route_score(trades)

        summary: Dict[str, Any] = {
            "symbols": per_symbol,
            "brokers": per_broker,
            "overall": overall,
            "route_score": route_score,
            "meta": {
                "num_trades": len(trades),
                "source_journal": str(self.journal_path),
                "source_ledger": str(self.ledger_path),
            },
        }

        self._save_summary_json(summary)

        return summary

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_journal_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.journal_path.exists():
            logger.warning("ExecutionAlpha: trade_journal.csv not found at %s", self.journal_path)
            return rows

        with self.journal_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        logger.info("ExecutionAlpha: Loaded %d trades from journal", len(rows))
        return rows

    def _load_ledger_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not self.ledger_path.exists():
            logger.info("ExecutionAlpha: order_ledger.jsonl not found at %s", self.ledger_path)
            return events

        with self.ledger_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    logger.exception("ExecutionAlpha: Failed to parse ledger line")
        logger.info("ExecutionAlpha: Loaded %d ledger events", len(events))
        return events

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def _analyze_per_symbol(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            sym = t.get("symbol") or "UNKNOWN"
            by_symbol.setdefault(sym, []).append(t)

        out: Dict[str, Any] = {}

        for sym, arr in by_symbol.items():
            slippages = []
            spreads = []
            lats = []
            pnls = []
            statuses = []

            for t in arr:
                sl = _safe_float(t.get("slippage"))
                sp = _safe_float(t.get("spread"))
                lat = _safe_float(t.get("latency_ms"))
                pnl = _safe_float(t.get("pnl"))
                status = t.get("status", "UNKNOWN")

                if sl is not None:
                    slippages.append(sl)
                if sp is not None:
                    spreads.append(sp)
                if lat is not None:
                    lats.append(lat)
                if pnl is not None:
                    pnls.append(pnl)
                statuses.append(status)

            total = len(arr)
            filled = sum(1 for s in statuses if s == "OK")
            blocked = sum(1 for s in statuses if s == "BLOCKED")
            errors = sum(1 for s in statuses if s == "ERROR")

            sym_summary = {
                "num_trades": total,
                "num_filled": filled,
                "num_blocked": blocked,
                "num_error": errors,
                "fill_rate": filled / total if total > 0 else None,
                "avg_slippage": mean(slippages) if slippages else None,
                "avg_spread": mean(spreads) if spreads else None,
                "avg_latency_ms": mean(lats) if lats else None,
            }

            if pnls:
                sym_summary.update(
                    {
                        "avg_pnl": mean(pnls),
                        "num_profitable": sum(1 for x in pnls if x > 0),
                        "num_unprofitable": sum(1 for x in pnls if x < 0),
                    }
                )

            out[sym] = sym_summary

        return out

    def _analyze_per_broker(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_broker: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            broker = t.get("broker") or "UNKNOWN"
            by_broker.setdefault(broker, []).append(t)

        out: Dict[str, Any] = {}

        for broker, arr in by_broker.items():
            slippages = []
            lats = []
            statuses = []
            scores = []

            for t in arr:
                sl = _safe_float(t.get("slippage"))
                lat = _safe_float(t.get("latency_ms"))
                sc = _safe_float(t.get("route_score"))
                status = t.get("status", "UNKNOWN")

                if sl is not None:
                    slippages.append(sl)
                if lat is not None:
                    lats.append(lat)
                if sc is not None:
                    scores.append(sc)
                statuses.append(status)

            total = len(arr)
            filled = sum(1 for s in statuses if s == "OK")
            blocked = sum(1 for s in statuses if s == "BLOCKED")
            errors = sum(1 for s in statuses if s == "ERROR")

            out[broker] = {
                "num_trades": total,
                "num_filled": filled,
                "num_blocked": blocked,
                "num_error": errors,
                "fill_rate": filled / total if total > 0 else None,
                "avg_slippage": mean(slippages) if slippages else None,
                "avg_latency_ms": mean(lats) if lats else None,
                "avg_route_score": mean(scores) if scores else None,
            }

        return out

    def _analyze_overall(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        slippages = []
        spreads = []
        lats = []
        pnls = []
        statuses = []

        for t in trades:
            sl = _safe_float(t.get("slippage"))
            sp = _safe_float(t.get("spread"))
            lat = _safe_float(t.get("latency_ms"))
            pnl = _safe_float(t.get("pnl"))
            status = t.get("status", "UNKNOWN")

            if sl is not None:
                slippages.append(sl)
            if sp is not None:
                spreads.append(sp)
            if lat is not None:
                lats.append(lat)
            if pnl is not None:
                pnls.append(pnl)
            statuses.append(status)

        total = len(trades)
        filled = sum(1 for s in statuses if s == "OK")
        blocked = sum(1 for s in statuses if s == "BLOCKED")
        errors = sum(1 for s in statuses if s == "ERROR")

        overall: Dict[str, Any] = {
            "num_trades": total,
            "num_filled": filled,
            "num_blocked": blocked,
            "num_error": errors,
            "fill_rate": filled / total if total > 0 else None,
            "avg_slippage": mean(slippages) if slippages else None,
            "avg_spread": mean(spreads) if spreads else None,
            "avg_latency_ms": mean(lats) if lats else None,
        }

        if pnls:
            overall.update(
                {
                    "avg_pnl": mean(pnls),
                    "num_profitable": sum(1 for x in pnls if x > 0),
                    "num_unprofitable": sum(1 for x in pnls if x < 0),
                }
            )

        return overall

    def _analyze_route_score(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple calibration: bucket trades by route_score and see slippage.

        Buckets:
            score >= 0.8
            0.5 <= score < 0.8
            score < 0.5
        """
        buckets = {
            "high": [],   # >= 0.8
            "mid": [],    # 0.5 - 0.8
            "low": [],    # < 0.5
            "missing": [],  # no score
        }

        for t in trades:
            score = _safe_float(t.get("route_score"))
            sl = _safe_float(t.get("slippage"))
            if sl is None:
                continue

            if score is None:
                buckets["missing"].append(sl)
            elif score >= 0.8:
                buckets["high"].append(sl)
            elif score >= 0.5:
                buckets["mid"].append(sl)
            else:
                buckets["low"].append(sl)

        out: Dict[str, Any] = {}
        for name, vals in buckets.items():
            if not vals:
                continue
            out[name] = {
                "num_trades": len(vals),
                "avg_slippage": mean(vals),
            }

        return out

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_summary_json(self, summary: Dict[str, Any]) -> None:
        SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        try:
            with SUMMARY_JSON.open("w") as f:
                json.dump(summary, f, indent=2)
            logger.info("ExecutionAlpha: Summary saved to %s", SUMMARY_JSON)
        except Exception:
            logger.exception("ExecutionAlpha: Failed to save summary JSON")
