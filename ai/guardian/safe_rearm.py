"""
ai/guardian/safe_rearm.py

Phase 96.1 – Safe Re-Arm Controller

Purpose
-------
After a kill-switch event (Phase 93/94/95), this module:

1. Verifies basic safety conditions:
   - Kill-switch is currently ACTIVE (unless overridden).
   - Account equity and buying power look sane.
   - Health snapshot global_error_count is below a configured threshold.
   - Position checks (optional: require flat book).

2. If conditions are acceptable:
   - Clears JSON kill-switch (ai.guardian.kill_switch.deactivate).
   - Optionally clears legacy trading_disabled.flag.
   - Optionally notifies via Telegram that the system is re-armed.
   - Optionally launches the Phase 95 Auto-Restart Supervisor.

This gives you a clean "SAFE-MODE EXIT" button you can run manually.
"""

from __future__ import annotations

import subprocess
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify
from ai.utils.alpaca_client import AlpacaClient
from ai.monitor.health_metrics import load_health_snapshot
from ai.guardian.kill_switch import status as kill_status, deactivate as deactivate_kill

ensure_env_loaded()
log = logging.getLogger("SafeRearm")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class RearmConfig:
    # Files
    legacy_flag_path: str = "data/runtime/trading_disabled.flag"

    # Safety checks
    require_kill_switch_active: bool = True
    min_equity: float = 1000.0   # minimal equity to consider the account "live"
    min_buying_power: float = 0.0
    max_global_error_count: int = 1000
    require_flat_book: bool = False  # if True, refuse to re-arm if there are any open positions

    # Supervisor auto-start
    auto_launch_supervisor: bool = False
    supervisor_module: str = "runner.phase95_supervisor"
    supervisor_config_path: str = "configs/phase95_supervisor.yaml"

    # Alerts
    telegram_enabled: bool = True
    telegram_tag: str = "Phase96.1Rearm"

    # Logging
    supervisor_log_to_console: bool = True


def load_rearm_config(path: str) -> RearmConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        log.warning("Rearm config not found at %s, using defaults", cfg_path)
        return RearmConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    legacy_flag_path = str(raw.get("legacy_flag_path", "data/runtime/trading_disabled.flag"))

    checks = raw.get("checks", {}) or {}
    require_kill_switch_active = bool(checks.get("require_kill_switch_active", True))
    min_equity = float(checks.get("min_equity", 1000.0))
    min_buying_power = float(checks.get("min_buying_power", 0.0))
    max_global_error_count = int(checks.get("max_global_error_count", 1000))
    require_flat_book = bool(checks.get("require_flat_book", False))

    sup = raw.get("supervisor", {}) or {}
    auto_launch_supervisor = bool(sup.get("auto_launch", False))
    supervisor_module = str(sup.get("module", "runner.phase95_supervisor"))
    supervisor_config_path = str(sup.get("config_path", "configs/phase95_supervisor.yaml"))
    supervisor_log_to_console = bool(sup.get("log_to_console", True))

    alerts = raw.get("alerts", {}) or {}
    telegram_enabled = bool(alerts.get("enabled", True))
    telegram_tag = str(alerts.get("tag", "Phase96.1Rearm"))

    return RearmConfig(
        legacy_flag_path=legacy_flag_path,
        require_kill_switch_active=require_kill_switch_active,
        min_equity=min_equity,
        min_buying_power=min_buying_power,
        max_global_error_count=max_global_error_count,
        require_flat_book=require_flat_book,
        auto_launch_supervisor=auto_launch_supervisor,
        supervisor_module=supervisor_module,
        supervisor_config_path=supervisor_config_path,
        telegram_enabled=telegram_enabled,
        telegram_tag=telegram_tag,
        supervisor_log_to_console=supervisor_log_to_console,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _summarize_account(broker: AlpacaClient) -> Dict[str, Any]:
    acct = broker.get_account()
    return {
        "equity": float(getattr(acct, "equity", 0.0)),
        "cash": float(getattr(acct, "cash", 0.0)),
        "buying_power": float(getattr(acct, "buying_power", 0.0)),
        "portfolio_value": float(getattr(acct, "portfolio_value", 0.0)),
        "multiplier": getattr(acct, "multiplier", None),
    }


def _summarize_positions(broker: AlpacaClient) -> Dict[str, Any]:
    positions = broker.get_positions()
    summary: Dict[str, Any] = {
        "total_positions": len(positions),
        "by_symbol": {},
    }
    for p in positions:
        try:
            sym = getattr(p, "symbol", "UNKNOWN")
            qty = float(getattr(p, "qty", 0.0))
            px = float(getattr(p, "current_price", 0.0))
            mv = qty * px
            summary["by_symbol"][sym] = {
                "qty": qty,
                "price": px,
                "market_value": mv,
                "side": getattr(p, "side", None),
            }
        except Exception:
            continue
    return summary


def _clear_legacy_flag(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
        log.info("Legacy flag removed: %s", path)
    except Exception as e:
        log.error("Failed to remove legacy flag %s: %s", path, e)


def _launch_supervisor(cfg: RearmConfig) -> None:
    cmd = [
        sys.executable,
        "-m",
        cfg.supervisor_module,
        "--config",
        cfg.supervisor_config_path,
    ]
    log.info("Launching supervisor: %s", " ".join(cmd))

    if cfg.supervisor_log_to_console:
        subprocess.Popen(cmd)
    else:
        # detach I/O if you want it silent; minimal handling here
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ---------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------
def perform_safety_checks(cfg: RearmConfig) -> Dict[str, Any]:
    """
    Run safety checks and return a dict with:
        {
          "ok": bool,
          "reasons": [str],
          "snapshot": { account, positions, health, kill_switch }
        }
    """
    broker = AlpacaClient()
    reasons: List[str] = []

    # Kill-switch status
    ks = kill_status()
    if cfg.require_kill_switch_active and not ks.get("kill", False):
        reasons.append("Kill-switch is not active, nothing to re-arm.")

    # Account
    try:
        acct = _summarize_account(broker)
    except Exception as e:
        acct = {"error": str(e)}
        reasons.append(f"Failed to load account: {e}")

    # Positions
    try:
        positions = _summarize_positions(broker)
    except Exception as e:
        positions = {"error": str(e)}
        reasons.append(f"Failed to load positions: {e}")

    # Health snapshot
    try:
        health = load_health_snapshot()
    except Exception as e:
        health = {"error": str(e)}
        reasons.append(f"Failed to load health snapshot: {e}")

    # Threshold checks (only if no errors above)
    if "error" not in acct:
        eq = acct.get("equity", 0.0) or 0.0
        bp = acct.get("buying_power", 0.0) or 0.0
        if eq < cfg.min_equity:
            reasons.append(
                f"Equity {eq:.2f} < min_equity {cfg.min_equity:.2f}"
            )
        if bp < cfg.min_buying_power:
            reasons.append(
                f"Buying power {bp:.2f} < min_buying_power {cfg.min_buying_power:.2f}"
            )

    if "error" not in health and isinstance(health, dict):
        ge = health.get("global_error_count")
        if isinstance(ge, int) and ge > cfg.max_global_error_count:
            reasons.append(
                f"global_error_count {ge} > max_global_error_count {cfg.max_global_error_count}"
            )

    if "error" not in positions and cfg.require_flat_book:
        total_pos = positions.get("total_positions", 0)
        if total_pos > 0:
            reasons.append(
                f"require_flat_book=True but total_positions={total_pos}"
            )

    ok = len(reasons) == 0

    snapshot = {
        "account": acct,
        "positions": positions,
        "health": health,
        "kill_switch": ks,
    }

    return {"ok": ok, "reasons": reasons, "snapshot": snapshot}


def rearm_system(cfg: RearmConfig, force: bool = False) -> int:
    """
    Perform safety checks, then (if safe or force=True):
        - Clear JSON kill-switch
        - Clear legacy flag
        - Optionally launch supervisor
        - Send Telegram summary

    Returns exit code: 0 on success, non-zero on failure/abort.
    """
    check = perform_safety_checks(cfg)
    ok = bool(check["ok"])
    reasons: List[str] = check["reasons"]
    snap: Dict[str, Any] = check["snapshot"]

    if not ok and not force:
        log.error("Safety checks FAILED; not re-arming.")
        for r in reasons:
            log.error("Reason: %s", r)

        # Telegram notification
        if cfg.telegram_enabled:
            try:
                msg = (
                    f"[{cfg.telegram_tag}] Re-arm aborted due to failed safety checks.\n"
                    + "\n".join(f"- {r}" for r in reasons[:10])
                )
                notify(msg, kind="guardian", meta={"snapshot": snap, "reasons": reasons})
            except Exception as e:
                log.error("Failed to send Telegram abort alert: %s", e)

        return 1

    if not ok and force:
        log.warning("Safety checks FAILED but force=True; proceeding anyway.")
        for r in reasons:
            log.warning("Reason (ignored due to force): %s", r)

    # Clear JSON kill-switch
    try:
        deactivate_kill()
        log.info("JSON kill-switch deactivated.")
    except Exception as e:
        log.error("Failed to deactivate JSON kill-switch: %s", e)
        return 2

    # Clear legacy flag
    _clear_legacy_flag(Path(cfg.legacy_flag_path))

    # Telegram notification - success
    if cfg.telegram_enabled:
        try:
            ks = snap.get("kill_switch") or {}
            msg = (
                f"[{cfg.telegram_tag}] System re-armed.\n"
                f"Previous kill flag: {ks}\n"
                f"Supervisor auto-launch: {cfg.auto_launch_supervisor}"
            )
            notify(msg, kind="guardian", meta=snap)
        except Exception as e:
            log.error("Failed to send Telegram success alert: %s", e)

    # Optionally start supervisor
    if cfg.auto_launch_supervisor:
        try:
            _launch_supervisor(cfg)
        except Exception as e:
            log.error("Failed to launch supervisor: %s", e)
            return 3

    log.info("Safe re-arm completed.")
    return 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 96.1 – Safe Re-Arm Controller"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/phase96_rearm.yaml",
        help="Path to Phase 96.1 re-arm YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-arm even if safety checks fail (NOT RECOMMENDED).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("=== Phase 96.1 – Safe Re-Arm Controller starting ===")
    cfg = load_rearm_config(args.config)
    rc = rearm_system(cfg, force=args.force)
    log.info("=== Phase 96.1 – Safe Re-Arm Controller finished (rc=%s) ===", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
