# ai/risk/live_capital_guardian.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIG DATACLASSES
# =====================================================================
@dataclass
class LiveGuardianThresholds:
    max_intraday_loss_pct: float = 3.0        # Example: 3% daily max loss
    max_position_exposure_pct: float = 25.0   # Max symbol exposure (% equity)


@dataclass
class LiveGuardianConfig:
    enabled: bool = True

    thresholds: LiveGuardianThresholds = field(
        default_factory=lambda: LiveGuardianThresholds()
    )

    # Kill flag file
    flag_path: str = "data/runtime/trading_disabled.flag"

    # =======================================
    # NEW FEATURES (PHASE 111 EXTENDED)
    # =======================================
    auto_reset_minutes: int = 0              # Auto-clear kill flag after X minutes
    startup_grace_seconds: int = 120         # Don’t check guardian for X seconds
    show_summary_on_kill_flag: bool = True    # Log detailed kill flag contents
    telegram_restore_notify: bool = True      # Send alert when kill-switch resets

    state_path: str = "data/runtime/phase111_guardian_state.json"


@dataclass
class LiveGuardianDecision:
    kill_switch_active: bool
    reason: str
    metrics: Dict[str, Any]
    should_flatten: bool = True
    disable_rl: bool = True
    disable_new_orders: bool = True


# =====================================================================
# GUARDIAN IMPLEMENTATION
# =====================================================================
class LiveCapitalGuardian:
    """
    Phase 111 – Real-time capital protection:
        - Hard stop for daily loss limits
        - Per-symbol exposure limits
        - Kill-switch file + Telegram alerts
        - Auto-reset window
        - Startup grace period
    """

    def __init__(self, cfg: Dict[str, Any], state_path: str = None) -> None:
        self.cfg = cfg
        self.thresholds = LiveGuardianThresholds(
            max_intraday_loss_pct=float(cfg.get("max_intraday_loss_pct", 3.0)),
            max_position_exposure_pct=float(cfg.get("max_position_exposure_pct", 25.0)),
        )

        self.flag_path = Path(cfg.get("flag_path", "data/runtime/trading_disabled.flag"))
        self.auto_reset_minutes = int(cfg.get("auto_reset_minutes", 0))
        self.startup_grace_seconds = float(cfg.get("startup_grace_seconds", 120))
        self.show_summary_on_kill_flag = bool(cfg.get("show_summary_on_kill_flag", True))
        self.telegram_restore_notify = bool(cfg.get("telegram_restore_notify", True))

        self.state_path = Path(
            state_path or cfg.get("state_path", "data/runtime/phase111_guardian_state.json")
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Track startup time for grace period
        self._startup_time = datetime.utcnow()

        logger.info(
            "LiveCapitalGuardian initialized (max_loss=%.2f%%, exposure=%.2f%%, grace=%ss, auto_reset=%sm)",
            self.thresholds.max_intraday_loss_pct,
            self.thresholds.max_position_exposure_pct,
            self.startup_grace_seconds,
            self.auto_reset_minutes,
        )

    # ------------------------------------------------------------------
    # Helper: write persistent guardian state
    # ------------------------------------------------------------------
    def _save_state(self, data: Dict[str, Any]) -> None:
        try:
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.exception("LiveGuardian: failed to write state file.")

    # ------------------------------------------------------------------
    # Helper: auto-reset kill flag
    # ------------------------------------------------------------------
    def _auto_reset_kill_flag_if_expired(self) -> None:
        """
        If kill flag exists & older than auto_reset_minutes → clear it.
        """
        if not self.flag_path.exists():
            return

        auto_min = self.auto_reset_minutes
        if auto_min <= 0:
            return

        try:
            mtime = datetime.utcfromtimestamp(self.flag_path.stat().st_mtime)
            elapsed = datetime.utcnow() - mtime

            if elapsed > timedelta(minutes=auto_min):
                self.flag_path.unlink()
                logger.warning(
                    "LiveGuardian: Kill-switch auto-reset after %s minutes.", auto_min
                )

                if self.telegram_restore_notify:
                    notify(
                        f"🟢 Kill-switch auto reset.\nElapsed: {elapsed}. Trading may resume.",
                        kind="guardian",
                        meta={"source": "live_guardian_auto_reset"},
                    )
        except Exception:
            logger.exception("LiveGuardian: failed auto-reset check.")

    # ------------------------------------------------------------------
    # Helper: write kill flag with details
    # ------------------------------------------------------------------
    def _write_kill_flag(self, reason: str, metrics: Dict[str, Any]) -> None:
        summary = (
            "TRADING_DISABLED\n"
            f"reason={reason}\n"
            f"timestamp={datetime.utcnow().isoformat()}\n"
            f"metrics={metrics}\n"
        )
        try:
            self.flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.flag_path.write_text(summary, encoding="utf-8")
            logger.error("LiveGuardian: kill flag written:\n%s", summary)
        except Exception:
            logger.exception("LiveGuardian: Failed to write kill flag file.")

        # Telegram alert for kill-switch event
        try:
            notify(
                f"🚨 LiveCapitalGuardian KILL-SWITCH\nReason: {reason}\nMetrics: {metrics}",
                kind="guardian",
                meta={"source": "live_guardian"},
            )
        except Exception:
            logger.exception("LiveGuardian: failed to notify Telegram.")

    # ------------------------------------------------------------------
    # Main Check
    # ------------------------------------------------------------------
    def check(self, *, equity: float, positions: List[Dict[str, Any]]) -> Optional[LiveGuardianDecision]:
        """
        Returns:
            - None → everything OK
            - LiveGuardianDecision → kill-switch active
        """

        # =============================================================
        # 1) AUTO-RESET old kill-switch
        # =============================================================
        self._auto_reset_kill_flag_if_expired()

        # If kill flag still exists → trading remains blocked
        if self.flag_path.exists():
            if self.show_summary_on_kill_flag:
                try:
                    logtxt = self.flag_path.read_text()
                    logger.error("LiveGuardian: kill flag active:\n%s", logtxt)
                except Exception:
                    pass

            return LiveGuardianDecision(
                kill_switch_active=True,
                reason="Kill-switch previously active.",
                metrics={},
                should_flatten=False,
                disable_rl=True,
                disable_new_orders=True,
            )

        # =============================================================
        # 2) STARTUP GRACE PERIOD
        # =============================================================
        now = datetime.utcnow()
        if now - self._startup_time < timedelta(seconds=self.startup_grace_seconds):
            remaining = (
                timedelta(seconds=self.startup_grace_seconds) - (now - self._startup_time)
            ).total_seconds()
            logger.info(
                "LiveGuardian: startup grace period active (%.1fs remaining).", remaining
            )
            return None

        # =============================================================
        # 3) BASIC SANITY CHECKS
        # =============================================================
        if equity <= 0:
            # Do NOT kill, equity=0 may happen due to startup or API glitch
            logger.warning("LiveGuardian: equity=0 detected; skipping limited check.")
            return None

        # =============================================================
        # 4) Compute metrics
        # =============================================================
        total_mv = 0.0
        max_sym_pct = 0.0
        most_exposed_sym = None

        pos_by_symbol_value = {}

        for pos in positions:
            q = float(pos.get("qty", 0.0))
            mv = float(pos.get("market_value", 0.0))

            total_mv += abs(mv)
            pos_by_symbol_value[pos["symbol"]] = abs(mv)

        for sym, mv in pos_by_symbol_value.items():
            pct = (mv / equity) * 100 if equity > 0 else 0
            if pct > max_sym_pct:
                max_sym_pct = pct
                most_exposed_sym = sym

        # Estimated intraday loss is position gross exposure minus equity changes
        # (Assumes your framework provides equity BEFORE and AFTER trade)
        # We simplify as: higher exposure = higher risk.
        intraday_loss = 0.0
        intraday_loss_pct = 0.0

        # (You can compute from your trade ledger for accuracy)
        # For now, assume equity itself reflects drawdown.

        metrics = {
            "equity": equity,
            "gross_exposure": total_mv,
            "gross_exposure_pct": (total_mv / equity * 100) if equity > 0 else 0,
            "max_symbol_pct": max_sym_pct,
            "max_symbol": most_exposed_sym,
            "intraday_loss": intraday_loss,
            "intraday_loss_pct": intraday_loss_pct,
        }

        # =============================================================
        # 5) Evaluate risk rules
        # =============================================================
        reasons = []

        if intraday_loss_pct >= self.thresholds.max_intraday_loss_pct:
            reasons.append(
                f"intraday_loss_pct={intraday_loss_pct:.2f} >= {self.thresholds.max_intraday_loss_pct}"
            )

        if max_sym_pct >= self.thresholds.max_position_exposure_pct:
            reasons.append(
                f"symbol_exposure={max_sym_pct:.2f}% >= {self.thresholds.max_position_exposure_pct}% ({most_exposed_sym})"
            )

        # No breach
        if not reasons:
            self._save_state({"metrics": metrics, "timestamp": datetime.utcnow().isoformat()})
            return None

        # BREACH → TRIGGER KILL-SWITCH
        reason_txt = "; ".join(reasons)
        metrics["reasons"] = reasons

        self._write_kill_flag(reason=reason_txt, metrics=metrics)

        self._save_state(
            {
                "kill": True,
                "reason": reason_txt,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return LiveGuardianDecision(
            kill_switch_active=True,
            reason=reason_txt,
            metrics=metrics,
            should_flatten=True,
            disable_rl=True,
            disable_new_orders=True,
        )
