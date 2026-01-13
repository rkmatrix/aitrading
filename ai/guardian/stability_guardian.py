"""
ai/guardian/stability_guardian.py

Phase 93 + 94 + 95 + 97 – Stability Guardian, Circuit Breaker Monitor,
Kill-Switch Bridge & Heartbeat Checker

Monitors:
    - Intraday equity loss
    - Historical drawdown (via equity history CSV)
    - Gross exposure vs equity
    - Error bursts via health_metrics.json (Phase 94)
    - Tick latency from Phase 26 (Phase 94)
    - Heartbeat staleness for Phase 26 (Phase 97)

On hard breach:
    - Sends Telegram alerts
    - Flips legacy trading_disabled.flag (for older phases)
    - Activates global JSON kill-switch (for Phase 95 Auto-Restart Supervisor)
"""

from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify
from ai.utils.alpaca_client import AlpacaClient
from ai.monitor.health_metrics import load_health_snapshot
from ai.guardian.kill_switch import activate as activate_kill, status as kill_status

ensure_env_loaded()
log = logging.getLogger("StabilityGuardian")


# ---------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------
@dataclass
class KillSwitchConfig:
    enabled: bool = True
    # Legacy flag (still written for backwards compatibility)
    flag_path: str = "data/runtime/trading_disabled.flag"


@dataclass
class GuardianConfig:
    check_interval_sec: int = 60
    max_intraday_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_gross_exposure_pct: float = 150.0
    equity_history_path: str = "data/reports/phase37_equity_history.csv"
    drawdown_lookback: int = 250

    # kill-switch
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)

    # alert controls
    alert_enabled: bool = True
    min_alert_interval_sec: int = 300

    # Phase 94 – error & latency monitoring
    error_monitor_enabled: bool = True
    max_error_burst: int = 5          # max new errors between checks
    max_tick_latency_sec: float = 5.0 # max Phase 26 tick duration

    # Phase 97 – integrated heartbeat monitoring
    heartbeat_enabled: bool = True
    heartbeat_path: str = "data/runtime/heartbeat_phase26.json"
    heartbeat_max_stale_sec: float = 30.0
    heartbeat_warn_stale_sec: float = 15.0


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_guardian_config(path: str) -> GuardianConfig:
    p = Path(path)
    if not p.exists():
        log.warning("Guardian config not found at %s, using defaults", path)
        return GuardianConfig()

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    ks_raw = raw.get("kill_switch", {}) or {}
    kill_switch = KillSwitchConfig(
        enabled=bool(ks_raw.get("enabled", True)),
        flag_path=str(ks_raw.get("flag_path", "data/runtime/trading_disabled.flag")),
    )

    alert_raw = raw.get("alert", {}) or {}
    alert_enabled = bool(alert_raw.get("enabled", True))
    min_alert_interval_sec = int(alert_raw.get("min_alert_interval_sec", 300))

    err_raw = raw.get("error_monitor", {}) or {}
    error_monitor_enabled = bool(err_raw.get("enabled", True))
    max_error_burst = int(err_raw.get("max_error_burst", 5))
    max_tick_latency_sec = float(err_raw.get("max_tick_latency_sec", 5.0))

    hb_raw = raw.get("heartbeat_monitor", {}) or {}
    heartbeat_enabled = bool(hb_raw.get("enabled", True))
    heartbeat_path = str(hb_raw.get("path", "data/runtime/heartbeat_phase26.json"))
    heartbeat_max_stale_sec = float(hb_raw.get("max_stale_sec", 30.0))
    heartbeat_warn_stale_sec = float(hb_raw.get("warn_stale_sec", 15.0))

    return GuardianConfig(
        check_interval_sec=int(raw.get("check_interval_sec", 60)),
        max_intraday_loss_pct=float(raw.get("max_intraday_loss_pct", 3.0)),
        max_drawdown_pct=float(raw.get("max_drawdown_pct", 15.0)),
        max_gross_exposure_pct=float(raw.get("max_gross_exposure_pct", 150.0)),
        equity_history_path=str(
            raw.get(
                "equity_history_path",
                "data/reports/phase37_equity_history.csv",
            )
        ),
        drawdown_lookback=int(raw.get("drawdown_lookback", 250)),
        kill_switch=kill_switch,
        alert_enabled=alert_enabled,
        min_alert_interval_sec=min_alert_interval_sec,
        error_monitor_enabled=error_monitor_enabled,
        max_error_burst=max_error_burst,
        max_tick_latency_sec=max_tick_latency_sec,
        heartbeat_enabled=heartbeat_enabled,
        heartbeat_path=heartbeat_path,
        heartbeat_max_stale_sec=heartbeat_max_stale_sec,
        heartbeat_warn_stale_sec=heartbeat_warn_stale_sec,
    )


# ---------------------------------------------------------------------
# StabilityGuardian
# ---------------------------------------------------------------------
class StabilityGuardian:
    def __init__(self, cfg: GuardianConfig) -> None:
        self.cfg = cfg
        self.broker = AlpacaClient()

        self._session_open_equity: Optional[float] = None
        self._last_alert_ts: Dict[str, float] = {}

        self._equity_history: Optional[pd.DataFrame] = None

        # Phase 94: previous error count snapshot
        self._prev_global_error_count: int = 0

    # ---------------------- helpers ----------------------
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
            log.info("Skipping alert '%s' due to throttle window.", key)
            return

        log.warning("⚠️ GUARDIAN ALERT [%s]: %s", key, msg)
        try:
            notify(msg, kind="guardian", meta=meta or {})
        except Exception as e:
            log.error("Failed to send Telegram alert: %s", e)

    def _flip_kill_switch(self, reason: str, meta: Dict[str, Any]) -> None:
        """
        Activates both:
        - Legacy text flag (trading_disabled.flag)
        - JSON kill-switch (used by Phase 95 Auto-Restart Supervisor)
        """
        if not self.cfg.kill_switch.enabled:
            return

        # 1) Legacy flag for older phases
        try:
            flag_path = Path(self.cfg.kill_switch.flag_path)
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_path.write_text(
                f"TRADING_DISABLED\nreason={reason}\nmeta={meta}\n",
                encoding="utf-8",
            )
            log.error("🚨 Kill switch flag written to %s (reason=%s)", flag_path, reason)
        except Exception as e:
            log.error("Failed to write legacy kill-switch flag: %s", e)

        # 2) JSON kill-switch for Phase 95 supervisor
        try:
            activate_kill(reason=reason)
            log.error("🚨 JSON kill-switch activated via ai.guardian.kill_switch (reason=%s)", reason)
        except Exception as e:
            log.error("Failed to activate JSON kill-switch: %s", e)

    # ---------------------- heartbeat loading ----------------------
    def _load_heartbeat(self) -> Dict[str, Any]:
        """
        Returns a dict:
            {
              "age_sec": float | None,
              "raw": dict | None,
            }
        If heartbeat is missing or invalid, age_sec will be None.
        """
        if not self.cfg.heartbeat_enabled:
            return {"age_sec": None, "raw": None}

        path = Path(self.cfg.heartbeat_path)
        if not path.exists():
            log.warning("Heartbeat file not found at %s", path)
            return {"age_sec": None, "raw": None}

        try:
            txt = path.read_text(encoding="utf-8")
            hb = json.loads(txt)
        except Exception as e:
            log.error("Failed to read/parse heartbeat at %s: %s", path, e)
            return {"age_sec": None, "raw": None}

        ts = float(hb.get("last_heartbeat_ts", 0.0))
        if ts <= 0:
            return {"age_sec": None, "raw": hb}

        age = self._now() - ts
        return {"age_sec": age, "raw": hb}

    # ---------------------- equity history ----------------------
    def _load_equity_history(self) -> None:
        p = Path(self.cfg.equity_history_path)
        if not p.exists():
            self._equity_history = None
            return

        try:
            df = pd.read_csv(p)
            if "equity" not in df.columns:
                log.warning("Equity history missing 'equity' column; skipping drawdown calc.")
                self._equity_history = None
                return

            self._equity_history = df.tail(self.cfg.drawdown_lookback).copy()
        except Exception as e:
            log.error("Failed to read equity history at %s: %s", p, e)
            self._equity_history = None

    def _compute_history_drawdown_pct(self, current_equity: float) -> float:
        if self._equity_history is None or self._equity_history.empty:
            return 0.0

        try:
            series = self._equity_history["equity"].astype(float)
            combined = list(series.values) + [current_equity]
            peak = max(combined)
            dd = (peak - current_equity) / max(peak, 1.0)
            return float(dd)
        except Exception as e:
            log.error("Error computing history drawdown: %s", e)
            return 0.0

    # ---------------------- main metrics ----------------------
    def _compute_metrics(self) -> Dict[str, Any]:
        acct = self.broker.get_account()
        equity = float(getattr(acct, "equity", 0.0))
        buying_power = float(getattr(acct, "buying_power", 0.0))

        if self._session_open_equity is None:
            self._session_open_equity = equity

        intraday_loss = self._session_open_equity - equity
        intraday_loss_pct = intraday_loss / max(self._session_open_equity, 1.0) * 100.0

        positions = self.broker.get_positions()
        gross_exposure = 0.0
        for p in positions:
            try:
                qty = float(getattr(p, "qty", 0.0))
                px = float(getattr(p, "current_price", 0.0))
                gross_exposure += abs(qty * px)
            except Exception:
                continue

        gross_exposure_pct = (
            gross_exposure / max(equity, 1.0) * 100.0 if equity > 0 else 0.0
        )

        self._load_equity_history()
        history_dd_frac = self._compute_history_drawdown_pct(equity)
        history_dd_pct = history_dd_frac * 100.0

        # Phase 94 – load health metrics
        health = load_health_snapshot()
        global_err = int(health.get("global_error_count", 0))
        lat = health.get("latency", {}) or {}
        tick_latency_sec = float(lat.get("phase26_tick_duration_sec", 0.0))

        # Phase 97 – heartbeat snapshot
        hb_info = self._load_heartbeat()
        hb_age = hb_info.get("age_sec")
        hb_raw = hb_info.get("raw")

        metrics = {
            "equity": equity,
            "buying_power": buying_power,
            "intraday_loss": intraday_loss,
            "intraday_loss_pct": intraday_loss_pct,
            "gross_exposure": gross_exposure,
            "gross_exposure_pct": gross_exposure_pct,
            "history_drawdown_pct": history_dd_pct,
            "position_count": len(positions),
            "global_error_count": global_err,
            "tick_latency_sec": tick_latency_sec,
            "heartbeat_age_sec": hb_age,
            "heartbeat_raw": hb_raw,
        }

        log.info(
            "Guardian metrics: Eq=%.2f | BP=%.2f | LossToday=%.2f (%.2f%%) | "
            "GrossEx=%.2f (%.1f%%) | DD_hist=%.2f%% | Pos=%d | "
            "ErrTotal=%d | TickLatency=%.3fs | HB_age=%s",
            equity,
            buying_power,
            intraday_loss,
            intraday_loss_pct,
            gross_exposure,
            gross_exposure_pct,
            history_dd_pct,
            len(positions),
            global_err,
            tick_latency_sec,
            f"{hb_age:.3f}s" if hb_age is not None else "n/a",
        )

        return metrics

    def _check_thresholds(self, m: Dict[str, Any]) -> None:
        hard_breach = False
        reasons = []

        # ----- Intraday loss -----
        if m["intraday_loss_pct"] >= self.cfg.max_intraday_loss_pct:
            reasons.append(
                f"intraday_loss_pct={m['intraday_loss_pct']:.2f} "
                f">= {self.cfg.max_intraday_loss_pct:.2f}"
            )
            self._send_alert(
                "intraday_loss",
                (
                    f"🚨 Intraday loss limit breached!\n"
                    f"Loss today: {m['intraday_loss']:.2f} "
                    f"({m['intraday_loss_pct']:.2f}%)\n"
                    f"Equity: {m['equity']:.2f}, BP: {m['buying_power']:.2f}"
                ),
                meta=m,
            )
            hard_breach = True

        # ----- Historical drawdown -----
        if m["history_drawdown_pct"] >= self.cfg.max_drawdown_pct:
            reasons.append(
                f"history_drawdown_pct={m['history_drawdown_pct']:.2f} "
                f">= {self.cfg.max_drawdown_pct:.2f}"
            )
            self._send_alert(
                "max_drawdown",
                (
                    f"⚠️ Max drawdown breached!\n"
                    f"History DD: {m['history_drawdown_pct']:.2f}%%\n"
                    f"Equity: {m['equity']:.2f}, BP: {m['buying_power']:.2f}"
                ),
                meta=m,
            )
            hard_breach = True

        # ----- Gross exposure -----
        if m["gross_exposure_pct"] >= self.cfg.max_gross_exposure_pct:
            reasons.append(
                f"gross_exposure_pct={m['gross_exposure_pct']:.2f} "
                f">= {self.cfg.max_gross_exposure_pct:.2f}"
            )
            self._send_alert(
                "gross_exposure",
                (
                    f"⚠️ Gross exposure too high!\n"
                    f"Gross: {m['gross_exposure']:.2f} "
                    f"({m['gross_exposure_pct']:.1f}%% of equity)\n"
                    f"Equity: {m['equity']:.2f}"
                ),
                meta=m,
            )
            hard_breach = True

        # ----- Phase 94: error bursts -----
        if self.cfg.error_monitor_enabled:
            current_err = int(m.get("global_error_count", 0))
            delta = max(0, current_err - self._prev_global_error_count)

            if delta >= self.cfg.max_error_burst and self.cfg.max_error_burst > 0:
                reasons.append(
                    f"error_burst={delta} >= {self.cfg.max_error_burst}"
                )
                self._send_alert(
                    "error_burst",
                    (
                        "🚨 Error burst detected!\n"
                        f"New errors since last check: {delta}\n"
                        f"Total global errors: {current_err}"
                    ),
                    meta=m,
                )
                hard_breach = True

            # update snapshot for next check
            self._prev_global_error_count = current_err

        # ----- Phase 94: high latency -----
        if self.cfg.max_tick_latency_sec > 0:
            tl = float(m.get("tick_latency_sec", 0.0))
            if tl >= self.cfg.max_tick_latency_sec:
                reasons.append(
                    f"tick_latency_sec={tl:.3f} >= {self.cfg.max_tick_latency_sec:.3f}"
                )
                self._send_alert(
                    "high_latency",
                    (
                        "⚠️ High tick latency from Phase 26!\n"
                        f"Observed: {tl:.3f} s\n"
                        f"Threshold: {self.cfg.max_tick_latency_sec:.3f} s"
                    ),
                    meta=m,
                )
                # treat persistent latency as hard breach too
                hard_breach = True

        # ----- Phase 97: heartbeat staleness -----
        if self.cfg.heartbeat_enabled:
            hb_age = m.get("heartbeat_age_sec")
            hb_raw = m.get("heartbeat_raw")

            if hb_age is None:
                # Missing or invalid heartbeat file -> treat as very serious
                reasons.append("heartbeat missing or unreadable")
                self._send_alert(
                    "heartbeat_missing",
                    "🚨 Heartbeat missing or unreadable in StabilityGuardian.",
                    meta={"heartbeat_raw": hb_raw},
                )
                hard_breach = True
            else:
                if hb_age >= self.cfg.heartbeat_max_stale_sec:
                    reasons.append(
                        f"heartbeat_age_sec={hb_age:.2f} >= "
                        f"{self.cfg.heartbeat_max_stale_sec:.2f}"
                    )
                    self._send_alert(
                        "heartbeat_stale",
                        (
                            "🚨 Heartbeat stale in StabilityGuardian!\n"
                            f"Age: {hb_age:.2f}s "
                            f"(max={self.cfg.heartbeat_max_stale_sec:.2f}s)"
                        ),
                        meta={"heartbeat_age_sec": hb_age, "heartbeat_raw": hb_raw},
                    )
                    hard_breach = True
                elif hb_age >= self.cfg.heartbeat_warn_stale_sec:
                    # Warning tier
                    self._send_alert(
                        "heartbeat_warn",
                        (
                            "⚠️ Heartbeat delayed in StabilityGuardian.\n"
                            f"Age: {hb_age:.2f}s "
                            f"(warn={self.cfg.heartbeat_warn_stale_sec:.2f}s)"
                        ),
                        meta={"heartbeat_age_sec": hb_age, "heartbeat_raw": hb_raw},
                    )

        if hard_breach and reasons:
            reason_str = "; ".join(reasons)
            meta = dict(m)
            meta["reasons"] = reasons
            self._flip_kill_switch(reason_str, meta)

    # ---------------------- public loop ----------------------
    def run_forever(self) -> None:
        log.info("🛡 StabilityGuardian started with cfg=%s", self.cfg)

        while True:
            try:
                # If kill-switch is already active, stop monitoring loop
                ks = kill_status()
                if ks.get("kill", False):
                    reason = ks.get("reason", "unknown")
                    self._send_alert(
                        "kill_switch_active",
                        f"🛑 Kill-switch already active (reason={reason}); "
                        "StabilityGuardian will stop monitoring.",
                        meta={"kill_switch": ks},
                    )
                    log.warning(
                        "Kill-switch active (reason=%s). Guardian stopping loop.", reason
                    )
                    break

                metrics = self._compute_metrics()
                self._check_thresholds(metrics)
            except Exception as e:
                log.error("Guardian loop error: %s", e)

            time.sleep(self.cfg.check_interval_sec)
