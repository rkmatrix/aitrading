from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

def _read_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

def append_audit(entry: Dict[str, Any], path: Path, max_records: int = 5000) -> None:
    """Append a single audit record; trim oldest if limit exceeded."""
    log = _read_json(path)
    entry.setdefault("timestamp", datetime.utcnow().isoformat())
    log.append(entry)
    if len(log) > max_records:
        log = log[-max_records:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")

def audit_from_phase_outputs(eval_json: Path, adj_json: Path, path: Path, max_records: int = 5000):
    """Read Phase 39/40 outputs and record a combined audit row."""
    try:
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        adj_data = json.loads(adj_json.read_text(encoding="utf-8"))
    except Exception:
        return
    grade = eval_data.get("grade", "N/A")
    metrics = eval_data.get("metrics", {}) or {}
    limits = adj_data.get("limits", {}) or {}
    row = {
        "grade": grade,
        "score": eval_data.get("score"),
        "drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "sortino": metrics.get("sortino"),
        "vol_ann": metrics.get("vol_ann"),
        "guardian_mode": (adj_data.get("guardian") or {}).get("mode"),
        "max_position_pct": limits.get("max_position_pct"),
        "gross_leverage": limits.get("gross_leverage"),
        "order_rate_per_min": limits.get("order_rate_per_min"),
        "source_eval": str(eval_json),
        "source_adj": str(adj_json),
    }
    append_audit(row, path, max_records)
