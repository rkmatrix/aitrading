# ai/monitor/heartbeat_writer.py
"""
Heartbeat writer for realtime components (Phase 97).

Used primarily by Phase 26 to record:
    - last heartbeat wall-clock time
    - tick sequence number
    - last tick duration
    - basic meta info (pid, component name, etc.)

The heartbeat file is then monitored by:
    - ai.monitor.heartbeat_monitor.HeartbeatMonitor (Phase 97 runner)
    - StabilityGuardian (Phase 93/97 integration)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("HeartbeatWriter")


@dataclass
class HeartbeatWriter:
    path: Path
    component: str = "phase26"
    pid: int = field(default_factory=os.getpid)

    tick_seq: int = 0

    def update(
        self,
        *,
        tick_duration_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Bump tick sequence and write a JSON heartbeat snapshot.
        """
        self.tick_seq += 1
        now = time.time()

        payload: Dict[str, Any] = {
            "component": self.component,
            "pid": self.pid,
            "tick_seq": self.tick_seq,
            "last_heartbeat_ts": now,
            "last_heartbeat_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(now)
            ),
        }

        if tick_duration_sec is not None:
            payload["last_tick_duration_sec"] = float(tick_duration_sec)

        if extra:
            payload["extra"] = extra

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to write heartbeat to %s: %s", self.path, e)
