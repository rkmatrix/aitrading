# ai/monitor/heartbeat_monitor.py
"""
Phase 97 – Heartbeat Monitor

Standalone watchdog that:
    - Reads heartbeat JSON (written by Phase 26).
    - Computes staleness (age of last heartbeat).
    - Checks last tick duration, tick sequence monotonicity, etc.
    - When a hard freeze is detected, activates global kill-switch.

This is *in addition* to StabilityGuardian's own heartbeat check (Option C).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore

from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify
from ai.guardian.kill_switch import activate as activate_kill, status as kill_status

ensure_env_loaded()
log = logging.getLogger("HeartbeatMonitor")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class HeartbeatMonitorConfig:
    heartbeat_path: str = "data/runtime/heartbeat_phase26.json"
    check_interval_sec: int = 10

    # Hard freeze if no heartbeat within this many seconds
    max_stale_sec: float = 30.0

    # Soft warning if heartbeats are slightly late
    warn_stale_sec: float = 15.0

    # Optional: if >0, treat very long last_tick_duration_sec as a problem
    max_tick_duration_sec: float = 0.0

    require_monotonic_seq: bool = True

    # Kill-switch integration
    kill_switch_enabled: bool = True

    # Alerts
    alert_enabled: bool = True
    min_alert_interval_sec: int = 120
    alert_tag: str = "Phase97Heartbeat"


def load_hb_monitor_config(path: str) -> HeartbeatMonitorConfig:
    p = Path(path)
    if not p.exists():
        log.warning("Heartbeat monitor config not found at %s, using defaults", path)
        return HeartbeatMonitorConfig()

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    hb_raw = raw.get("heartbeat", {}) or {}
    alerts = raw.get("alerts", {}) or {}

    return HeartbeatMonitorConfig(
        heartbeat_path=str(hb_raw.get("path", "data/runtime/heartbeat_phase26.json")),
        check_interval_sec=int(hb_raw.get("check_interval_sec", 10)),
        max_stale_sec=float(hb_raw.get("max_stale_sec", 30.0)),
        warn_stale_sec=float(hb_raw.get("warn_stale_sec", 15.0)),
        max_tick_duration_sec=float(hb_raw.get("max_tick_duration_sec", 0.0)),
        require_monotonic_seq=bool(hb_raw.get("require_monotonic_seq", True)),
        kill_switch_enabled=bool(raw.get("kill_switch_enabled", True)),
        alert_enabled=bool(alerts.get("enabled", True)),
        min_alert_interval_sec=int(alerts.get("min_alert_interval_sec", 120)),
        alert_tag=str(alerts.get("tag", "Phase97Heartbeat")),
    )


# ---------------------------------------------------------------------
# HeartbeatMonitor
# ---------------------------------------------------------------------
class HeartbeatMonitor:
    def __init__(self, cfg: HeartbeatMonitorConfig) -> None:
        self.cfg = cfg
        self._last_alert_ts: Dict[str, float] = {}
        self._last_tick_seq: Optional[int] = None

    def _now(self) -> float:
        return time.time()

    def _should_alert(self, key: str) -> bool:
        if not self.cfg.alert_enabled:
            return False
        now = self._now()
        last = self._last_alert_ts.get(key)
        if last is None or (now - last) >= self.cfg.min_alert_interval_sec:
            self._last_alert_ts[key] = now
            return True
        return False

    def _send_alert(self, key: str, msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self._should_alert(key):
            log.info("Skipping heartbeat alert '%s' due to throttle window.", key)
            return

        log.warning("⚠️ HEARTBEAT ALERT [%s]: %s", key, msg)
        try:
            notify(msg, kind="guardian", meta=meta or {})
        except Exception as e:
            log.error("Failed to send Telegram alert: %s", e)

    def _load_heartbeat(self) -> Optional[Dict[str, Any]]:
        path = Path(self.cfg.heartbeat_path)
        if not path.exists():
            return None

        try:
            txt = path.read_text(encoding="utf-8")
            return json.loads(txt)
        except Exception as e:
            log.error("Failed to read/parse heartbeat file %s: %s", path, e)
            return None

    def _flip_kill_switch(self, reason: str, meta: Dict[str, Any]) -> None:
        if not self.cfg.kill_switch_enabled:
            return

        try:
            activate_kill(reason=reason)
            self._send_alert("heartbeat_kill", f"🛑 Kill-switch activated: {reason}", meta)
            log.error("Kill-switch activated by HeartbeatMonitor (reason=%s)", reason)
        except Exception as e:
            log.error("Failed to activate kill-switch from HeartbeatMonitor: %s", e)

    def check_once(self) -> None:
        # If kill-switch is already active, no need to continue
        ks = kill_status()
        if ks.get("kill", False):
            log.warning(
                "Kill-switch already active (reason=%s); HeartbeatMonitor will idle.",
                ks.get("reason", "n/a"),
            )
            return

        hb = self._load_heartbeat()
        now = self._now()

        if hb is None:
            reason = "No heartbeat file present."
            self._flip_kill_switch(reason, {"now": now})
            return

        ts = float(hb.get("last_heartbeat_ts", 0.0))
        age = now - ts if ts > 0 else 1e9
        tick_seq = hb.get("tick_seq")
        last_tick_duration = hb.get("last_tick_duration_sec")

        meta = {
            "age_sec": age,
            "tick_seq": tick_seq,
            "last_tick_duration_sec": last_tick_duration,
            "raw": hb,
        }

        # Staleness
        if age >= self.cfg.max_stale_sec:
            reason = (
                f"Heartbeat stale: age={age:.2f}s >= max_stale_sec={self.cfg.max_stale_sec:.2f}s"
            )
            self._flip_kill_switch(reason, meta)
            return

        # Warning tier
        if age >= self.cfg.warn_stale_sec:
            self._send_alert(
                "heartbeat_warn",
                (
                    f"Heartbeat delayed: age={age:.2f}s "
                    f"(warn_stale_sec={self.cfg.warn_stale_sec:.2f}s)"
                ),
                meta,
            )

        # Monotonic sequence
        if self.cfg.require_monotonic_seq and isinstance(tick_seq, int):
            if self._last_tick_seq is not None and tick_seq <= self._last_tick_seq:
                reason = (
                    f"Non-monotonic heartbeat tick_seq: {tick_seq} "
                    f"(last={self._last_tick_seq})"
                )
                self._flip_kill_switch(reason, meta)
                return
            self._last_tick_seq = tick_seq

        # Tick duration
        if (
            self.cfg.max_tick_duration_sec > 0
            and last_tick_duration is not None
            and last_tick_duration > self.cfg.max_tick_duration_sec
        ):
            self._send_alert(
                "tick_duration_warn",
                (
                    "High last_tick_duration detected in heartbeat: "
                    f"{last_tick_duration:.3f}s (max={self.cfg.max_tick_duration_sec:.3f}s)"
                ),
                meta,
            )

        log.info(
            "Heartbeat OK: age=%.3fs, tick_seq=%s, last_tick_duration=%s",
            age,
            tick_seq,
            last_tick_duration,
        )

    def run_forever(self) -> None:
        log.info("🫀 HeartbeatMonitor started with cfg=%s", self.cfg)
        while True:
            try:
                self.check_once()
            except Exception as e:
                log.error("Heartbeat monitor loop error: %s", e)
            time.sleep(self.cfg.check_interval_sec)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 97 – Heartbeat Monitor")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/phase97_heartbeat.yaml",
        help="Path to Phase 97 heartbeat monitor YAML config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_hb_monitor_config(args.config)
    mon = HeartbeatMonitor(cfg)
    mon.run_forever()


if __name__ == "__main__":
    main()
