# ai/analytics/decision_loader.py
"""
Phase 64 – Decision Loader Utilities

Helpers to:
    • Load Phase 63 decision CSV summary
    • Load Phase 63 JSONL detailed records (with votes, context, order_meta)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_decision_csv(path: str | Path) -> pd.DataFrame:
    """
    Load the Phase 63 CSV summary file.

    Columns expected (from DecisionRecorder):
        ts, mode, symbol, final_action, final_size, final_broker,
        fused_conf, conflict_score, num_votes, order_sent, order_status
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Decision CSV not found: {p}")
    df = pd.read_csv(p)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df


def load_decision_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load the detailed decision log from JSONL.

    Returns:
        List of dicts with keys:
            ts, mode, symbol, context, decision, order_sent, order_meta, extra
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Decision JSONL not found: {p}")

    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return records


def decisions_jsonl_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert JSONL records (detailed) into a flattened DataFrame.

    For dashboard use we extract:
        ts, mode, symbol, final_action, fused_conf, conflict_score,
        final_size, final_broker, order_sent, order_status, num_votes
    """
    flat_rows: List[Dict[str, Any]] = []
    for r in records:
        d = r.get("decision", {}) or {}
        votes = d.get("votes", []) or []
        flat_rows.append(
            {
                "ts": r.get("ts"),
                "mode": r.get("mode"),
                "symbol": r.get("symbol"),
                "final_action": d.get("final_action"),
                "final_size": d.get("final_size"),
                "final_broker": d.get("final_broker"),
                "fused_conf": d.get("fused_conf"),
                "conflict_score": d.get("conflict_score"),
                "num_votes": len(votes),
                "order_sent": r.get("order_sent"),
                "order_status": (r.get("order_meta") or {}).get("status"),
                "raw_record": r,
            }
        )

    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df


def match_record(
    jsonl_records: List[Dict[str, Any]],
    ts: str,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Try to find a JSONL record that matches a CSV row by ts + symbol.

    ts should be ISO or string convertible to datetime; comparison is
    done on exact string match after ISO formatting where possible.
    """
    # Normalize ts to a simple string prefix to avoid microsecond issues
    ts_str = str(ts).split("+")[0]  # strip timezone for matching
    for r in jsonl_records:
        r_ts = str(r.get("ts", "")).split("+")[0]
        if r_ts == ts_str and str(r.get("symbol")) == str(symbol):
            return r
    return None


__all__ = [
    "load_decision_csv",
    "load_decision_jsonl",
    "decisions_jsonl_to_df",
    "match_record",
]
