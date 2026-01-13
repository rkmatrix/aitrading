from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("FillExporter")


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    # generic object
    d = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        # keep it JSON-ish
        if isinstance(v, (str, int, float, bool)) or v is None:
            d[k] = v
    return d


def export_fills_to_jsonl(
    fills: Iterable[Any],
    out_path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a list/iterable of fill objects (dict/dataclass/object) into JSONL replay file.

    Each line:
      {"ts": "...", "type": "fill", "fill": {...}, "meta": {...}}
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for x in fills:
            # Phase56-compatible JSONL (one raw event per line)
            payload = _to_dict(x)
            
            # Add minimal metadata if missing
            payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
            payload.setdefault("event", "fill")
            
            f.write(json.dumps(payload) + "\n")

            n += 1

    log.info("Exported %d fills → %s", n, out_path)
    return out_path


def default_replay_path(prefix: str = "fills") -> str:
    d = datetime.now().strftime("%Y-%m-%d")
    return str(Path("data") / "replay" / f"{d}_{prefix}.jsonl")
