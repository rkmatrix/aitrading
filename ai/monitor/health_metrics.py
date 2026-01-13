# ai/monitor/health_metrics.py

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("HealthMetrics")

HEALTH_METRICS_PATH = Path(
    os.getenv("HEALTH_METRICS_PATH", "data/runtime/health_metrics.json")
)


def _default_struct() -> Dict[str, Any]:
    return {
        "version": 1,
        "last_updated": 0.0,
        "global_error_count": 0,
        "sources": {},
        "latency": {},
    }


def _load_raw() -> Dict[str, Any]:
    if not HEALTH_METRICS_PATH.exists():
        return _default_struct()

    try:
        with HEALTH_METRICS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_struct()
        return data
    except Exception as e:
        log.error("Failed to read health metrics file %s: %s", HEALTH_METRICS_PATH, e)
        return _default_struct()


def _save_raw(data: Dict[str, Any]) -> None:
    try:
        HEALTH_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = HEALTH_METRICS_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(HEALTH_METRICS_PATH)
    except Exception as e:
        log.error("Failed to write health metrics file %s: %s", HEALTH_METRICS_PATH, e)


def record_error(source: str, kind: str, message: str) -> None:
    """
    Increment error counters for a given source and record the last error.

    source: e.g. "phase26_tick"
    kind:   e.g. "exception", "router_error"
    """
    try:
        data = _load_raw()
        sources = data.setdefault("sources", {})
        entry = sources.setdefault(source, {
            "error_count": 0,
        })

        entry["error_count"] = int(entry.get("error_count", 0)) + 1
        entry["last_error_kind"] = kind
        entry["last_error"] = str(message)
        entry["last_error_ts"] = time.time()

        data["global_error_count"] = int(data.get("global_error_count", 0)) + 1
        data["last_updated"] = time.time()

        _save_raw(data)
    except Exception as e:
        log.error("record_error failed: %s", e)


def record_latency(source: str, metric: str, value: float) -> None:
    """
    Record a latency metric, e.g.:

        record_latency("phase26_tick", "duration_sec", 0.42)
    """
    try:
        data = _load_raw()
        latency = data.setdefault("latency", {})
        key = f"{source}_{metric}"
        latency[key] = float(value)
        latency["last_update_ts"] = time.time()
        data["last_updated"] = time.time()
        _save_raw(data)
    except Exception as e:
        log.error("record_latency failed: %s", e)


def load_health_snapshot() -> Dict[str, Any]:
    """
    Load the current health metrics snapshot for readers (Guardian, dashboards).
    """
    return _load_raw()
