from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from .heartbeat_monitor import HeartbeatWriter

logger = logging.getLogger(__name__)


class FreezeLevel(Enum):
    OK = auto()
    WARNING = auto()
    RESTART_SUBSYSTEM = auto()
    RESTART_FULL = auto()
    FATAL = auto()


@dataclass
class FreezeConfig:
    heartbeat_timeout_sec: int = 5
    freeze_restart_sec: int = 20
    full_restart_sec: int = 60
    fatal_freeze_sec: int = 300


@dataclass
class FreezeDecision:
    level: FreezeLevel
    age_sec: Optional[float]
    message: str


class FreezeDetector:
    def __init__(self, cfg: FreezeConfig, heartbeat_path: str | Path) -> None:
        self.cfg = cfg
        self.heartbeat_path = Path(heartbeat_path)

    def evaluate(self) -> FreezeDecision:
        age = HeartbeatWriter.age_seconds(self.heartbeat_path)
        if age is None:
            return FreezeDecision(
                level=FreezeLevel.WARNING,
                age_sec=None,
                message=f"No heartbeat file yet at {self.heartbeat_path}",
            )

        msg = f"Heartbeat age={age:.1f}s (timeout={self.cfg.heartbeat_timeout_sec}s)"
        logger.debug(msg)

        if age < self.cfg.heartbeat_timeout_sec:
            return FreezeDecision(level=FreezeLevel.OK, age_sec=age, message=msg)

        if age < self.cfg.freeze_restart_sec:
            return FreezeDecision(
                level=FreezeLevel.WARNING,
                age_sec=age,
                message=f"Heartbeat delayed ({age:.1f}s) – monitoring",
            )

        if age < self.cfg.full_restart_sec:
            return FreezeDecision(
                level=FreezeLevel.RESTART_SUBSYSTEM,
                age_sec=age,
                message=f"Heartbeat stale ({age:.1f}s) – restart realtime subsystem",
            )

        if age < self.cfg.fatal_freeze_sec:
            return FreezeDecision(
                level=FreezeLevel.RESTART_FULL,
                age_sec=age,
                message=f"Heartbeat frozen ({age:.1f}s) – full system restart",
            )

        return FreezeDecision(
            level=FreezeLevel.FATAL,
            age_sec=age,
            message=f"Heartbeat dead ({age:.1f}s) – fatal freeze",
        )
