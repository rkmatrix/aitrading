from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta

def now_ts() -> float:
    return time.time()

def to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
