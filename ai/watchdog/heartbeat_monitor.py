from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Heartbeat:
    timestamp: float
    process: str
    tick: Optional[int] = None
    latency_ms: Optional[float] = None
    note: str = ""
    extra: Dict[str, Any] | None = None


class HeartbeatWriter:
    """
    Simple JSON heartbeat writer.

    Usage (inside realtime loop):

        hb = HeartbeatWriter("data/runtime/phase68_heartbeat.json", process="Phase26Realtime")
        ...
        while running:
            start = time.time()
            # do_tick()
            latency = (time.time() - start) * 1000
            hb.beat(tick=tick_idx, latency_ms=latency, note="tick")

    """

    def __init__(self, path: str | Path, *, process: str, mkdirs: bool = True) -> None:
        self.path = Path(path)
        self.process = process
        if mkdirs:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(
        self,
        *,
        tick: Optional[int] = None,
        latency_ms: Optional[float] = None,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        hb = Heartbeat(
            timestamp=time.time(),
            process=self.process,
            tick=tick,
            latency_ms=latency_ms,
            note=note,
            extra=extra or None,
        )
        try:
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(asdict(hb), indent=2))
            tmp_path.replace(self.path)
        except Exception:
            logger.exception("Failed to write heartbeat to %s", self.path)

    @staticmethod
    def read(path: str | Path) -> Optional[Heartbeat]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return Heartbeat(**data)
        except Exception:
            logger.exception("Failed to read/parse heartbeat from %s", p)
            return None

    @staticmethod
    def age_seconds(path: str | Path) -> Optional[float]:
        hb = HeartbeatWriter.read(path)
        if hb is None:
            return None
        return max(0.0, time.time() - hb.timestamp)
