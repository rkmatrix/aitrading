from __future__ import annotations
import json, logging, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml  # type: ignore

from .freeze_detector import FreezeConfig, FreezeDetector, FreezeLevel
from .heartbeat_monitor import HeartbeatWriter
from .subsystem_pinger import SubsystemPinger

try:
    from tools.telegram_alerts import notify  # type: ignore
except Exception:
    notify = None  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------
# FIX: All dataclasses now use default_factory
# -----------------------------

@dataclass
class TargetConfig:
    type: str = "module"
    command: str = "runner.phase26_realtime_live"
    args: List[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    heartbeat_timeout_sec: int = 5
    freeze_restart_sec: int = 20
    full_restart_sec: int = 60
    fatal_freeze_sec: int = 300
    alpaca_max_failures: int = 3


@dataclass
class RestartPolicy:
    max_restart_attempts: int = 5
    restart_window_sec: int = 600


@dataclass
class Phase68Config:
    mode: str = "PAPER"
    target: TargetConfig = field(default_factory=TargetConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    restart_policy: RestartPolicy = field(default_factory=RestartPolicy)

    heartbeat_path: str = "data/runtime/phase68_heartbeat.json"
    health_path: str = "data/runtime/phase68_health.json"
    crash_report_path: str = "data/reports/phase68_crash_report.json"

    enable_telegram: bool = True
    telegram_tag: str = "phase68_stability"


# ---------------------------------------------------------
# AUTO-RESTART ENGINE
# ---------------------------------------------------------

class AutoRestartEngine:
    def __init__(self, cfg_path: str | Path) -> None:
        self.cfg_path = Path(cfg_path)
        self.cfg = self._load_config(self.cfg_path)

        self.heartbeat_path = Path(self.cfg.heartbeat_path)
        self.health_path = Path(self.cfg.health_path)
        self.crash_report_path = Path(self.cfg.crash_report_path)

        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        self.crash_report_path.parent.mkdir(parents=True, exist_ok=True)

        self.hb_writer = HeartbeatWriter(self.heartbeat_path, process="Phase68Watchdog")

        self.freeze_detector = FreezeDetector(
            FreezeConfig(
                heartbeat_timeout_sec=self.cfg.thresholds.heartbeat_timeout_sec,
                freeze_restart_sec=self.cfg.thresholds.freeze_restart_sec,
                full_restart_sec=self.cfg.thresholds.full_restart_sec,
                fatal_freeze_sec=self.cfg.thresholds.fatal_freeze_sec,
            ),
            heartbeat_path=self.heartbeat_path,
        )

        self.pinger = SubsystemPinger(health_log_path=self.health_path)

        self.proc: Optional[subprocess.Popen] = None
        self.restart_history: List[float] = []
        self.alpaca_fail_count: int = 0

    # ---------------------------------------------------------
    # LOAD CONFIG
    # ---------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> Phase68Config:
        if not path.exists():
            raise FileNotFoundError(f"Phase68 config not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        target_raw = data.get("target", {})
        thresholds_raw = data.get("thresholds", {})
        restart_raw = data.get("restart_policy", {})

        cfg = Phase68Config(
            mode=data.get("mode", "PAPER"),

            target=TargetConfig(
                type=str(target_raw.get("type", "module")),
                command=str(target_raw.get("command", "runner.phase26_realtime_live")),
                args=list(target_raw.get("args", [])),
            ),

            thresholds=ThresholdConfig(
                heartbeat_timeout_sec=int(thresholds_raw.get("heartbeat_timeout_sec", 5)),
                freeze_restart_sec=int(thresholds_raw.get("freeze_restart_sec", 20)),
                full_restart_sec=int(thresholds_raw.get("full_restart_sec", 60)),
                fatal_freeze_sec=int(thresholds_raw.get("fatal_freeze_sec", 300)),
                alpaca_max_failures=int(thresholds_raw.get("alpaca_max_failures", 3)),
            ),

            restart_policy=RestartPolicy(
                max_restart_attempts=int(restart_raw.get("max_restart_attempts", 5)),
                restart_window_sec=int(restart_raw.get("restart_window_sec", 600)),
            ),

            heartbeat_path=str(data.get("heartbeat_path", "data/runtime/phase68_heartbeat.json")),
            health_path=str(data.get("health_path", "data/runtime/phase68_health.json")),
            crash_report_path=str(data.get("crash_report_path", "data/reports/phase68_crash_report.json")),
            enable_telegram=bool(data.get("enable_telegram", True)),
            telegram_tag=str(data.get("telegram_tag", "phase68_stability")),
        )

        return cfg

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------

    def run(self) -> None:
        logger.info("🚀 Phase68 AutoRestartEngine starting (mode=%s)…", self.cfg.mode)

        if self.cfg.enable_telegram:
            self._send_alert("system", "🚀 Phase68 stability engine starting", {"mode": self.cfg.mode})

        try:
            self._ensure_process()

            while True:
                loop_start = time.time()

                self.hb_writer.beat(note="watchdog_tick")

                self._check_process()
                self._check_subsystems()
                self._check_freeze()

                elapsed = time.time() - loop_start
                time.sleep(max(1.0, 3.0 - elapsed))

        except KeyboardInterrupt:
            logger.info("Phase68 interrupted, shutting down…")
            self._terminate_process()

        except Exception:
            logger.exception("Phase68 FATAL crash")
            self._record_crash("watchdog_crash")
            raise

    # ---------------------------------------------------------
    # PROCESS CONTROL
    # ---------------------------------------------------------

    def _build_command(self) -> List[str]:
        t = self.cfg.target

        if t.type == "module":
            cmd = [sys.executable, "-m", t.command]
            if t.args:
                cmd.extend(t.args)
            return cmd

        if t.type == "script":
            cmd = [sys.executable, t.command]
            if t.args:
                cmd.extend(t.args)
            return cmd

        return [t.command, *t.args]

    def _ensure_process(self) -> None:
        if self.proc and self.proc.poll() is None:
            return

        cmd = self._build_command()
        logger.info("🔁 Starting supervised process: %s", " ".join(cmd))

        self.proc = subprocess.Popen(cmd)
        self._record_restart("start")

    def _terminate_process(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return

        try:
            logger.info("🛑 Terminating supervised process (pid=%s)…", self.proc.pid)
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process stuck — force killing…")
            self.proc.kill()
        finally:
            self.proc = None

    def _restart_process(self, reason: str) -> None:
        if not self._can_restart():
            logger.error("🚫 Restart limit reached (%s)", reason)
            self._send_alert("system", f"🚫 Restart limit reached ({reason})")
            return

        logger.warning("🔁 Restarting supervised process (%s)", reason)
        self._send_alert("system", f"🔁 Restarting process ({reason})")

        self._terminate_process()
        time.sleep(2)
        self._ensure_process()

    def _check_process(self) -> None:
        if not self.proc:
            self._ensure_process()
            return

        code = self.proc.poll()
        if code is not None:
            logger.error("💥 Supervised process exited with code=%s", code)
            self._record_crash("process_exit", {"code": code})
            self._restart_process(f"exit_{code}")

    # ---------------------------------------------------------
    # SUBSYSTEM HEALTH
    # ---------------------------------------------------------

    def _check_subsystems(self) -> None:
        results = self.pinger.run_all()

        for r in results:
            if r.name == "alpaca_rest":
                if r.status == "ok":
                    self.alpaca_fail_count = 0
                else:
                    self.alpaca_fail_count += 1
                    logger.warning(
                        "Alpaca REST fail_count=%s err=%s",
                        self.alpaca_fail_count,
                        r.error,
                    )
                    if self.alpaca_fail_count >= self.cfg.thresholds.alpaca_max_failures:
                        self._restart_process("alpaca_unhealthy")
                        self.alpaca_fail_count = 0

    # ---------------------------------------------------------
    # FREEZE DETECTOR
    # ---------------------------------------------------------

    def _check_freeze(self) -> None:
        decision = self.freeze_detector.evaluate()

        if decision.level == FreezeLevel.OK:
            return

        logger.warning("FreezeDetector: %s", decision.message)

        if decision.level == FreezeLevel.WARNING:
            return

        if decision.level == FreezeLevel.RESTART_SUBSYSTEM:
            self._restart_process("freeze_restart")

        elif decision.level == FreezeLevel.RESTART_FULL:
            self._restart_process("full_restart")

        elif decision.level == FreezeLevel.FATAL:
            self._send_alert("system", "💀 FATAL freeze detected", {"detail": decision.message})
            self._restart_process("fatal_freeze")

    # ---------------------------------------------------------
    # RESTART POLICY
    # ---------------------------------------------------------

    def _record_restart(self, reason: str) -> None:
        now = time.time()
        self.restart_history.append(now)

        cutoff = now - self.cfg.restart_policy.restart_window_sec
        self.restart_history = [t for t in self.restart_history if t >= cutoff]

    def _can_restart(self) -> bool:
        now = time.time()
        cutoff = now - self.cfg.restart_policy.restart_window_sec
        self.restart_history = [t for t in self.restart_history if t >= cutoff]

        return len(self.restart_history) < self.cfg.restart_policy.max_restart_attempts

    # ---------------------------------------------------------
    # CRASH RECORDING & NOTIFICATIONS
    # ---------------------------------------------------------

    def _record_crash(self, kind: str, extra: Optional[dict] = None) -> None:
        payload = {"kind": kind, "timestamp": time.time(), "extra": extra or {}}

        try:
            with self.crash_report_path.open("w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            logger.exception("Failed to write crash report")

    def _send_alert(self, kind: str, msg: str, meta: Optional[dict] = None) -> None:
        if not self.cfg.enable_telegram or notify is None:
            return

        try:
            full_meta = {"tag": self.cfg.telegram_tag, "mode": self.cfg.mode, **(meta or {})}
            notify(msg, kind=kind, meta=full_meta)
        except Exception:
            logger.exception("Failed to send Telegram alert")
