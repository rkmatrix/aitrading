"""
ai/guardian/auto_recovery_wizard.py

Phase 96 – Auto-Recovery Wizard

Goals:
    - Inspect current kill-switch status (JSON + legacy flag).
    - Summarize Alpaca account & positions (risk exposure).
    - Load last health snapshot (errors, latency, etc).
    - Tail recent log files for quick context.
    - Print a human-readable recovery report.
    - (Optional) Notify via Telegram that a recovery report was generated.

This does NOT automatically clear the kill-switch.
Run it manually when you're in "SAFE-MODE HALT" and want a diagnosis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify
from ai.utils.alpaca_client import AlpacaClient
from ai.monitor.health_metrics import load_health_snapshot
from ai.guardian.kill_switch import status as kill_status

ensure_env_loaded()
log = logging.getLogger("AutoRecoveryWizard")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class LogFileConfig:
    path: str
    tail_lines: int = 100


@dataclass
class RecoveryConfig:
    # Legacy kill flag path (for completeness)
    legacy_flag_path: str = "data/runtime/trading_disabled.flag"

    # Log files to tail in the report
    log_files: List[LogFileConfig] = field(
        default_factory=lambda: [
            LogFileConfig(path="logs/phase26_realtime.log", tail_lines=80),
            LogFileConfig(path="logs/phase93_stability_guardian.log", tail_lines=80),
        ]
    )

    # Controls which checks to run
    check_account: bool = True
    check_positions: bool = True
    check_health_snapshot: bool = True

    # Telegram
    telegram_enabled: bool = True
    telegram_tag: str = "Phase96Recovery"


def load_recovery_config(path: str) -> RecoveryConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        log.warning("Recovery config not found at %s, using defaults", cfg_path)
        return RecoveryConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    legacy_flag_path = str(
        raw.get("legacy_flag_path", "data/runtime/trading_disabled.flag")
    )

    # log files
    logs_raw = raw.get("log_files", []) or []
    log_files: List[LogFileConfig] = []
    for item in logs_raw:
        if not isinstance(item, dict):
            continue
        log_files.append(
            LogFileConfig(
                path=str(item.get("path", "")),
                tail_lines=int(item.get("tail_lines", 100)),
            )
        )

    checks = raw.get("checks", {}) or {}
    check_account = bool(checks.get("account", True))
    check_positions = bool(checks.get("positions", True))
    check_health_snapshot = bool(checks.get("health_snapshot", True))

    alerts = raw.get("alerts", {}) or {}
    telegram_enabled = bool(alerts.get("enabled", True))
    telegram_tag = str(alerts.get("tag", "Phase96Recovery"))

    return RecoveryConfig(
        legacy_flag_path=legacy_flag_path,
        log_files=log_files or RecoveryConfig().log_files,
        check_account=check_account,
        check_positions=check_positions,
        check_health_snapshot=check_health_snapshot,
        telegram_enabled=telegram_enabled,
        telegram_tag=telegram_tag,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _tail_file(path: Path, n: int) -> List[str]:
    if not path.exists():
        return [f"<missing file: {path}>"]

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]
    except Exception as e:
        return [f"<error reading {path}: {e}>"]


def summarize_account(broker: AlpacaClient) -> Dict[str, Any]:
    acct = broker.get_account()
    out = {
        "equity": float(getattr(acct, "equity", 0.0)),
        "cash": float(getattr(acct, "cash", 0.0)),
        "buying_power": float(getattr(acct, "buying_power", 0.0)),
        "multiplier": getattr(acct, "multiplier", None),
        "portfolio_value": float(getattr(acct, "portfolio_value", 0.0)),
        "regt_buying_power": float(getattr(acct, "regt_buying_power", 0.0))
        if hasattr(acct, "regt_buying_power")
        else None,
    }
    return out


def summarize_positions(broker: AlpacaClient) -> Dict[str, Any]:
    positions = broker.get_positions()
    summary: Dict[str, Any] = {
        "total_positions": len(positions),
        "by_symbol": {},
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
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

            summary["gross_exposure"] += abs(mv)
            summary["net_exposure"] += mv
        except Exception:
            continue

    return summary


def load_legacy_flag(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return {"exists": True, "content": txt}
    except Exception as e:
        return {"exists": True, "error": str(e)}


# ---------------------------------------------------------------------
# Core: Recovery Report
# ---------------------------------------------------------------------
def generate_recovery_report(cfg: RecoveryConfig) -> Dict[str, Any]:
    broker = AlpacaClient()

    # 1) Kill-switch status
    ks = kill_status()
    legacy = load_legacy_flag(Path(cfg.legacy_flag_path))

    report: Dict[str, Any] = {
        "kill_switch": ks,
        "legacy_flag": legacy,
        "account": None,
        "positions": None,
        "health_snapshot": None,
        "logs": {},
        "suggestions": [],
    }

    # 2) Account & positions
    if cfg.check_account:
        try:
            report["account"] = summarize_account(broker)
        except Exception as e:
            report["account"] = {"error": str(e)}

    if cfg.check_positions:
        try:
            report["positions"] = summarize_positions(broker)
        except Exception as e:
            report["positions"] = {"error": str(e)}

    # 3) Health snapshot (Phase 94 metrics)
    if cfg.check_health_snapshot:
        try:
            report["health_snapshot"] = load_health_snapshot()
        except Exception as e:
            report["health_snapshot"] = {"error": str(e)}

    # 4) Tail logs
    for lf in cfg.log_files:
        path = Path(lf.path)
        report["logs"][str(path)] = _tail_file(path, lf.tail_lines)

    # 5) Suggestions (very lightweight, based on status)
    suggestions: List[str] = []

    if ks.get("kill", False):
        reason = ks.get("reason", "unknown")
        suggestions.append(
            f"Kill-switch is ACTIVE (reason={reason}). "
            "Investigate the metrics & logs below before clearing it."
        )
    else:
        suggestions.append(
            "Kill-switch is NOT active. If the bot is still halted, check legacy flag, "
            "health metrics, and Phase 26 logs."
        )

    acct = report.get("account") or {}
    if isinstance(acct, dict) and "equity" in acct and "buying_power" in acct:
        eq = acct.get("equity") or 0.0
        bp = acct.get("buying_power") or 0.0
        if bp <= 0:
            suggestions.append(
                "Buying power is zero or negative. Check for unsettled trades, "
                "margin issues, or broker restrictions."
            )
        if eq <= 0:
            suggestions.append(
                "Equity is non-positive. Verify account status in Alpaca dashboard."
            )

    health = report.get("health_snapshot") or {}
    if isinstance(health, dict):
        ge = health.get("global_error_count")
        if isinstance(ge, int) and ge > 0:
            suggestions.append(
                f"global_error_count={ge}. Inspect recent errors in your logs "
                "and fix root causes before re-enabling trading."
            )
        lat = health.get("latency", {}) or {}
        tl = lat.get("phase26_tick_duration_sec")
        if tl is not None and tl > 0:
            suggestions.append(
                f"phase26_tick_duration_sec≈{tl}. If this is high, investigate performance "
                "bottlenecks or external API slowness."
            )

    report["suggestions"] = suggestions
    return report


def _format_report_for_console(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 80)
    lines.append("PHASE 96 – AUTO-RECOVERY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Kill-switch
    ks = report.get("kill_switch") or {}
    lines.append("Kill-Switch Status:")
    lines.append(f"  kill:   {ks.get('kill', False)}")
    lines.append(f"  reason: {ks.get('reason', 'n/a')}")
    lines.append("")

    # Legacy flag
    legacy = report.get("legacy_flag") or {}
    lines.append("Legacy trading_disabled.flag:")
    lines.append(f"  exists: {legacy.get('exists', False)}")
    content = legacy.get("content")
    err = legacy.get("error")
    if content:
        lines.append("  content (first 5 lines):")
        for l in str(content).splitlines()[:5]:
            lines.append(f"    {l}")
    if err:
        lines.append(f"  error reading flag: {err}")
    lines.append("")

    # Account
    acct = report.get("account")
    lines.append("Alpaca Account Summary:")
    if isinstance(acct, dict) and "error" not in acct:
        for k in [
            "equity",
            "cash",
            "buying_power",
            "portfolio_value",
            "multiplier",
            "regt_buying_power",
        ]:
            if k in acct and acct[k] is not None:
                lines.append(f"  {k}: {acct[k]}")
    else:
        lines.append(f"  <unavailable> {acct}")
    lines.append("")

    # Positions
    pos = report.get("positions")
    lines.append("Positions Summary:")
    if isinstance(pos, dict) and "error" not in pos:
        lines.append(f"  total_positions: {pos.get('total_positions')}")
        lines.append(f"  gross_exposure:  {pos.get('gross_exposure')}")
        lines.append(f"  net_exposure:    {pos.get('net_exposure')}")
        by_sym = pos.get("by_symbol") or {}
        lines.append("  by_symbol (up to 10):")
        for i, (sym, info) in enumerate(by_sym.items()):
            if i >= 10:
                lines.append("    ...")
                break
            lines.append(
                f"    {sym}: qty={info.get('qty')}, px={info.get('price')}, "
                f"mv={info.get('market_value')}, side={info.get('side')}"
            )
    else:
        lines.append(f"  <unavailable> {pos}")
    lines.append("")

    # Health snapshot
    hs = report.get("health_snapshot")
    lines.append("Health Snapshot (Phase 94):")
    if isinstance(hs, dict) and "error" not in hs:
        ge = hs.get("global_error_count")
        lines.append(f"  global_error_count: {ge}")
        lat = hs.get("latency") or {}
        for k, v in lat.items():
            lines.append(f"  latency[{k}]: {v}")
    else:
        lines.append(f"  <unavailable> {hs}")
    lines.append("")

    # Logs
    lines.append("Log Tails:")
    logs = report.get("logs") or {}
    for path, tail in logs.items():
        lines.append("-" * 80)
        lines.append(f"Log: {path}")
        lines.append("-" * 80)
        if isinstance(tail, list):
            for l in tail:
                lines.append(l)
        else:
            lines.append(str(tail))
        lines.append("")

    # Suggestions
    lines.append("=" * 80)
    lines.append("SUGGESTED NEXT STEPS:")
    lines.append("=" * 80)
    for s in report.get("suggestions", []):
        lines.append(f"- {s}")

    return "\n".join(lines)


def run_recovery(cfg: RecoveryConfig) -> None:
    report = generate_recovery_report(cfg)
    text = _format_report_for_console(report)

    # Print to console
    print(text)

    # Telegram notification (just a short summary, not full log dump)
    if cfg.telegram_enabled:
        try:
            ks = report.get("kill_switch") or {}
            reason = ks.get("reason", "n/a")
            kill_flag = ks.get("kill", False)
            msg = (
                f"[{cfg.telegram_tag}] Recovery report generated.\n"
                f"kill={kill_flag}, reason={reason}\n"
                f"See console / logs for full details."
            )
            notify(msg, kind="guardian", meta={"kill_switch": ks})
        except Exception as e:
            log.error("Failed to send recovery Telegram alert: %s", e)


# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 96 – Auto-Recovery Wizard"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/phase96_recovery.yaml",
        help="Path to Phase 96 recovery YAML config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("=== Phase 96 – Auto-Recovery Wizard starting ===")
    cfg = load_recovery_config(args.config)
    run_recovery(cfg)
    log.info("=== Phase 96 – Auto-Recovery Wizard finished ===")


if __name__ == "__main__":
    main()
