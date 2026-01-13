# ai/monitor/execution_health_monitor.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)


@dataclass
class ExecutionHealthConfig:
    """
    Phase 114 – Execution Health Monitor

    Thresholds are conservative: if we see too many errors in a short period,
    we prefer to fail safe (flatten + stop trading) instead of silently limping.
    """
    max_error_ticks: int = 3           # number of recent error ticks before we kill
    warn_ticks_since_trade: int = 0    # reserved for future (not used now)
    flatten_on_unhealthy: bool = True
    write_kill_flag: bool = True
    kill_flag_path: str = "data/runtime/trading_disabled.flag"
    component_name: str = "phase26_realtime"
    telegram_kind: str = "guardian"


@dataclass
class ExecutionHealthDecision:
    healthy: bool
    reason: Optional[str]
    should_flatten: bool
    should_stop_trading: bool


class ExecutionHealthMonitor:
    """
    Simple execution health monitor for Phase 26.

    We intentionally keep it simple and explainable:

        - If error_ticks >= max_error_ticks:
              → unhealthy, flatten, stop, write kill flag, Telegram.
        - Otherwise:
              → healthy.

    More complex metrics (heartbeat age, API latencies) can be added later.
    """

    def __init__(self, cfg: ExecutionHealthConfig) -> None:
        self.cfg = cfg
        self.kill_flag_path = Path(cfg.kill_flag_path)

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------
    def check(
        self,
        *,
        error_ticks: int,
        ticks_since_trade: int,
    ) -> ExecutionHealthDecision:
        """
        Evaluate the current execution health based on counters maintained by Phase 26.
        """

        # Primary hard condition: too many error ticks
        if error_ticks >= self.cfg.max_error_ticks:
            reason = (
                f"Execution unhealthy: error_ticks={error_ticks} "
                f"(>= {self.cfg.max_error_ticks})."
            )
            logger.error("ExecutionHealthMonitor: %s", reason)
            return ExecutionHealthDecision(
                healthy=False,
                reason=reason,
                should_flatten=self.cfg.flatten_on_unhealthy,
                should_stop_trading=True,
            )

        # For now, ticks_since_trade is only informational / future use
        return ExecutionHealthDecision(
            healthy=True,
            reason=None,
            should_flatten=False,
            should_stop_trading=False,
        )

    # ------------------------------------------------------------------
    # Kill-switch side effects
    # ------------------------------------------------------------------
    def raise_kill_switch(self, reason: str) -> None:
        """
        Write kill-switch flag and send Telegram alert.
        Phase 26 will see this via _check_kill_switch() and stop.
        """
        msg = (
            f"🚨 Phase114 Execution Health KILL-SWITCH\n"
            f"Component: {self.cfg.component_name}\n"
            f"Reason: {reason}\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}"
        )

        if self.cfg.write_kill_flag:
            try:
                self.kill_flag_path.parent.mkdir(parents=True, exist_ok=True)
                self.kill_flag_path.write_text(msg, encoding="utf-8")
                logger.error(
                    "ExecutionHealthMonitor: kill flag written to %s",
                    self.kill_flag_path,
                )
            except Exception:
                logger.exception(
                    "ExecutionHealthMonitor: failed to write kill flag to %s",
                    self.kill_flag_path,
                )

        try:
            notify(msg, kind=self.cfg.telegram_kind, meta={"source": "phase114"})
        except Exception:
            logger.exception(
                "ExecutionHealthMonitor: failed to send Telegram alert."
            )
