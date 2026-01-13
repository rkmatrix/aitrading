# ai/guardian/auto_restart_supervisor.py
"""
Auto-Restart & Self-Healing Supervisor (Phase 95)

Responsibilities
----------------
- Launch and monitor Phase 26 realtime loop (or any target module).
- Restart the child process when it exits unexpectedly (crash).
- Respect a global kill-switch (env var and/or file-based).
- Throttle restarts to avoid infinite crash loops.
- Optionally send Telegram alerts when restarts or kill-events happen.

Usage
-----
Supervisor is normally driven by runner/phase95_supervisor.py.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("AutoRestartSupervisor")

try:
    # Optional, do not hard-crash if not available
    from tools.telegram_alerts import notify as tg_notify  # type: ignore
except Exception:  # pragma: no cover
    tg_notify = None


@dataclass
class SupervisorConfig:
    target_module: str = "runner.phase26_realtime_live"
    target_config: str = "configs/phase26_realtime.yaml"

    # restart policy
    max_restarts: int = 10
    restart_window_seconds: int = 3600  # time window for max_restarts
    cooldown_seconds: int = 15          # wait before restarting after crash
    restart_on_clean_exit: bool = False # if False: no restart when exit code == 0

    # kill switch
    kill_switch_file: Optional[str] = "data/runtime/kill_switch.json"
    kill_switch_env: Optional[str] = "AITB_KILL_SWITCH"
    kill_grace_seconds: int = 15

    # alerts
    enable_telegram: bool = True
    alert_tag: str = "Phase95Supervisor"

    # logging
    log_child_output: bool = False  # if True, pipe child stdout/stderr


@dataclass
class AutoRestartSupervisor:
    cfg: SupervisorConfig
    restart_times: List[float] = field(default_factory=list)
    _child: Optional[subprocess.Popen] = None

    def _send_alert(self, msg: str, kind: str = "system") -> None:
        logger.info(msg)
        if self.cfg.enable_telegram and tg_notify is not None:
            try:
                tg_notify(
                    f"[{self.cfg.alert_tag}] {msg}",
                    kind=kind,
                    meta={"tag": self.cfg.alert_tag},
                )
            except Exception:
                logger.exception("Failed to send Telegram alert")

    # ---------- Kill-Switch Handling ----------

    def _kill_switch_from_env(self) -> bool:
        if not self.cfg.kill_switch_env:
            return False
        val = os.getenv(self.cfg.kill_switch_env, "").strip().lower()
        return val in {"1", "true", "yes", "on", "kill", "stop"}

    def _kill_switch_from_file(self) -> bool:
        if not self.cfg.kill_switch_file:
            return False
        path = Path(self.cfg.kill_switch_file)
        if not path.exists():
            return False

        # If the file exists but is not valid JSON, be conservative and treat as ON.
        try:
            data = json.loads(path.read_text().strip() or "{}")
        except Exception:
            return True

        # Allow a couple of simple patterns
        flag = str(data.get("kill") or data.get("enabled") or data.get("status", "")).lower()
        if isinstance(data, dict) and data.get("kill") is True:
            return True
        if flag in {"1", "true", "yes", "on", "kill", "stop", "halt"}:
            return True
        return False

    def kill_switch_active(self) -> bool:
        return self._kill_switch_from_env() or self._kill_switch_from_file()

    # ---------- Child Process Management ----------

    def _spawn_child(self) -> subprocess.Popen:
        """Spawn the Phase 26 process (or configured target)."""
        cmd = [
            sys.executable,
            "-m",
            self.cfg.target_module,
            "--config",
            self.cfg.target_config,
        ]

        logger.info("🚀 Launching child process: %s", " ".join(cmd))

        if self.cfg.log_child_output:
            proc = subprocess.Popen(cmd)
        else:
            # inherit stdout/stderr so logs appear in same console
            proc = subprocess.Popen(cmd)

        self._child = proc
        return proc

    def _terminate_child(self, reason: str = "kill-switch") -> None:
        if not self._child:
            return

        proc = self._child
        if proc.poll() is not None:
            return  # already stopped

        self._send_alert(f"Terminating child process due to {reason} (pid={proc.pid})")

        try:
            # First try a graceful termination
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.terminate()
        except Exception:
            logger.exception("Failed to send terminate signal to child")

        # Wait for process to exit gracefully
        start = time.time()
        while proc.poll() is None and (time.time() - start) < self.cfg.kill_grace_seconds:
            time.sleep(0.5)

        if proc.poll() is None:
            logger.warning("Child did not exit gracefully; killing it")
            try:
                proc.kill()
            except Exception:
                logger.exception("Failed to kill child process")

    # ---------- Restart Throttling ----------

    def _record_restart(self) -> None:
        now = time.time()
        self.restart_times.append(now)

        # prune old entries outside restart_window_seconds
        window_start = now - self.cfg.restart_window_seconds
        self.restart_times = [t for t in self.restart_times if t >= window_start]

    def _too_many_restarts(self) -> bool:
        if self.cfg.max_restarts <= 0:
            return False
        return len(self.restart_times) > self.cfg.max_restarts

    # ---------- Main Loop ----------

    def run_forever(self) -> None:
        """
        Main supervisor loop.

        - Check kill-switch before each start and after each crash.
        - Restart only on unexpected (non-zero) exit, unless restart_on_clean_exit is True.
        - Exit if kill-switch is active or restart limit is exceeded.
        """
        self._send_alert("Phase 95 Auto-Restart Supervisor starting…")

        try:
            while True:
                if self.kill_switch_active():
                    logger.warning("Kill-switch is ACTIVE before launch; supervisor will not start child.")
                    self._send_alert("Kill-switch active; supervisor exiting without starting child.", kind="guardian")
                    break

                # Spawn child
                proc = self._spawn_child()

                # Wait for the child to exit
                try:
                    exit_code = proc.wait()
                except KeyboardInterrupt:
                    logger.info("Supervisor received KeyboardInterrupt, terminating child and exiting.")
                    self._terminate_child(reason="KeyboardInterrupt")
                    raise

                self._child = None
                logger.warning("Child exited with code %s", exit_code)

                # If kill-switch has been turned on while child was running, stop everything
                if self.kill_switch_active():
                    self._send_alert(
                        f"Kill-switch is now ACTIVE; not restarting child (last exit_code={exit_code}).",
                        kind="guardian",
                    )
                    break

                # Decide whether to restart
                crashed = exit_code not in (0, None)
                if not crashed and not self.cfg.restart_on_clean_exit:
                    logger.info(
                        "Child exited cleanly (exit_code=%s) and restart_on_clean_exit=False; supervisor exiting.",
                        exit_code,
                    )
                    break

                reason = "crash" if crashed else "clean exit (restart_on_clean_exit=True)"
                self._send_alert(f"Child terminated ({reason}); scheduling restart after cooldown…", kind="system")

                # throttle restarts
                self._record_restart()
                if self._too_many_restarts():
                    msg = (
                        f"Too many restarts ({len(self.restart_times)} within "
                        f"{self.cfg.restart_window_seconds}s); giving up."
                    )
                    logger.error(msg)
                    self._send_alert(msg, kind="guardian")
                    break

                # Cooldown before restart
                time.sleep(self.cfg.cooldown_seconds)

        finally:
            # Make sure we don't leak a child process if something strange happens
            if self._child is not None and self._child.poll() is None:
                self._terminate_child(reason="supervisor shutdown")
            self._send_alert("Phase 95 Auto-Restart Supervisor stopped.")
