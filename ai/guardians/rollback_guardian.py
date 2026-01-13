"""
Phase 47 – Rollback Guardian
----------------------------

Purpose:
    Watch key health metrics (equity, drawdown, error flags) and decide
    whether the system should roll back to a safer state.

Used by:
    • runner/phase47_rollback_guardian.py
    • runner/phase70_unified_live.py  (Phase 70 engine)

Public API:
    RollbackGuardian.from_yaml(path: str) -> RollbackGuardian
    RollbackGuardian.check() -> dict

`check()` returns a dict of the form:

    {
        "should_rollback": bool,
        "reasons": [ "STRING_REASON_1", "STRING_REASON_2", ... ],
        "checks": {
            "equity": {...},
            "errors": {...}
        }
    }

This module is defensive:
    • If CSV / error files are missing → logs warning, but does not crash.
    • If config values are missing → uses safe defaults.
"""

from __future__ import annotations
import logging
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("RollbackGuardian")


# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------
@dataclass
class RollbackGuardianConfig:
    # Path to equity history CSV (Phase 37 or Phase 70)
    # Expected columns: timestamp,equity,buying_power
    equity_history_path: str = "data/reports/phase70_equity_history.csv"

    # How many recent rows to look at for calculations
    equity_lookback_rows: int = 200

    # Trigger rollback if peak-to-last equity drawdown >= this
    max_drawdown_pct: float = 15.0

    # Trigger rollback if *latest* equity drop from previous row >= this
    max_single_step_drop_pct: float = 10.0

    # Optional error flag file: JSON with structure like:
    #    {"critical_errors": 3, "last_error": "..."}
    error_flag_path: str = "data/runtime/phase47_errors.json"

    # Trigger rollback if critical_errors >= this threshold
    max_critical_errors: int = 3

    # Enable/disable check types
    enable_equity_checks: bool = True
    enable_error_checks: bool = True

    # If True, guardian will still trigger `should_rollback`, but higher-level
    # logic may interpret this as "soft" (advisory) rather than hard.
    dry_run: bool = True

    # Additional free-form metadata
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# RollbackGuardian class
# ----------------------------------------------------------------------
class RollbackGuardian:
    def __init__(self, cfg: RollbackGuardianConfig) -> None:
        self.cfg = cfg
        self.log = logging.getLogger("RollbackGuardian")
        self.log.info(
            "RollbackGuardian initialized with equity_history_path=%s, error_flag_path=%s",
            cfg.equity_history_path,
            cfg.error_flag_path,
        )

    # ------------------------------------------------------------------
    # Factory from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "RollbackGuardian":
        """
        Create a RollbackGuardian from a YAML configuration file.

        YAML may look like:
            equity_history_path: data/reports/phase70_equity_history.csv
            equity_lookback_rows: 250
            max_drawdown_pct: 20.0
            max_single_step_drop_pct: 12.0
            error_flag_path: data/runtime/phase47_errors.json
            max_critical_errors: 5
            enable_equity_checks: true
            enable_error_checks: true
            dry_run: false
        Or can be nested under a top-level key (e.g. `rollback_guardian:`).
        """
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(
                "RollbackGuardian config YAML not found at %s; using defaults.",
                path,
            )
            return cls(RollbackGuardianConfig())

        with path_obj.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Allow nested structure like {"rollback_guardian": {...}}
        if "rollback_guardian" in raw and isinstance(raw["rollback_guardian"], dict):
            cfg_dict = raw["rollback_guardian"]
        else:
            cfg_dict = raw

        # Filter only known fields
        cfg_fields = {field.name for field in RollbackGuardianConfig.__dataclass_fields__.values()}
        filtered: Dict[str, Any] = {k: v for k, v in cfg_dict.items() if k in cfg_fields}

        # Build config
        cfg = RollbackGuardianConfig(**filtered)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Public check method
    # ------------------------------------------------------------------
    def check(self) -> Dict[str, Any]:
        """
        Perform all configured checks and return a structured result dict.

        Returns:
            {
                "should_rollback": bool,
                "reasons": [...],
                "checks": {
                    "equity": {...},
                    "errors": {...},
                }
            }
        """
        reasons: List[str] = []
        checks: Dict[str, Any] = {}

        # Equity / drawdown checks
        if self.cfg.enable_equity_checks:
            eq_result = self._check_equity()
            checks["equity"] = eq_result
            if eq_result.get("triggered"):
                reasons.append(eq_result.get("reason", "Equity check triggered."))

        # Error flag checks
        if self.cfg.enable_error_checks:
            err_result = self._check_errors()
            checks["errors"] = err_result
            if err_result.get("triggered"):
                reasons.append(err_result.get("reason", "Error check triggered."))

        should_rollback = len(reasons) > 0

        result = {
            "should_rollback": should_rollback,
            "reasons": reasons,
            "checks": checks,
            "dry_run": self.cfg.dry_run,
        }

        if should_rollback:
            self.log.warning("🛡️ RollbackGuardian triggered: %s", result)
        else:
            self.log.info("🛡️ RollbackGuardian check OK (no rollback).")

        return result

    # ------------------------------------------------------------------
    # Equity / Drawdown checks
    # ------------------------------------------------------------------
    def _check_equity(self) -> Dict[str, Any]:
        """
        Inspect recent equity history CSV and compute:
            - peak to last drawdown %
            - latest step drop %
        """
        path = Path(self.cfg.equity_history_path)
        if not path.exists():
            self.log.warning(
                "Equity history file missing at %s; skipping equity checks.",
                path,
            )
            return {
                "available": False,
                "triggered": False,
                "reason": "Equity history file missing.",
            }

        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            self.log.exception("Failed to read equity history CSV: %s", e)
            return {
                "available": False,
                "triggered": False,
                "reason": f"Error reading equity history: {e}",
            }

        if not rows:
            return {
                "available": False,
                "triggered": False,
                "reason": "Equity history file empty.",
            }

        # Use only the last N rows
        rows = rows[-self.cfg.equity_lookback_rows :]

        try:
            equities = [float(r["equity"]) for r in rows if "equity" in r]
        except Exception:
            self.log.exception("Failed to parse equity values from CSV.")
            return {
                "available": False,
                "triggered": False,
                "reason": "Could not parse equity values.",
            }

        if not equities:
            return {
                "available": False,
                "triggered": False,
                "reason": "No equity values in history.",
            }

        latest = equities[-1]
        peak = max(equities)
        dd_pct = 0.0
        if peak > 0:
            dd_pct = (peak - latest) / peak * 100.0

        # Latest step change (from previous point)
        if len(equities) >= 2:
            prev = equities[-2]
            if prev > 0:
                step_drop_pct = (prev - latest) / prev * 100.0
            else:
                step_drop_pct = 0.0
        else:
            step_drop_pct = 0.0

        triggered = False
        reason = "Equity checks OK."

        if dd_pct >= self.cfg.max_drawdown_pct:
            triggered = True
            reason = (
                f"Max drawdown exceeded: {dd_pct:.2f}% ≥ {self.cfg.max_drawdown_pct:.2f}%."
            )
        elif step_drop_pct >= self.cfg.max_single_step_drop_pct:
            triggered = True
            reason = (
                f"Single-step equity drop exceeded: {step_drop_pct:.2f}% ≥ "
                f"{self.cfg.max_single_step_drop_pct:.2f}%."
            )

        result = {
            "available": True,
            "triggered": triggered,
            "reason": reason,
            "latest_equity": latest,
            "peak_equity": peak,
            "drawdown_pct": dd_pct,
            "step_drop_pct": step_drop_pct,
            "thresholds": {
                "max_drawdown_pct": self.cfg.max_drawdown_pct,
                "max_single_step_drop_pct": self.cfg.max_single_step_drop_pct,
            },
        }
        return result

    # ------------------------------------------------------------------
    # Error flag checks
    # ------------------------------------------------------------------
    def _check_errors(self) -> Dict[str, Any]:
        """
        Inspect a JSON error flag file to see if critical error threshold
        has been exceeded.

        Expected JSON structure (but flexible):
            {
                "critical_errors": 3,
                "last_error": "some message",
                "last_timestamp": "2025-11-20T12:34:56"
            }
        """
        path = Path(self.cfg.error_flag_path)
        if not path.exists():
            return {
                "available": False,
                "triggered": False,
                "reason": "Error flag file missing.",
            }

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log.exception("Failed to parse error flag JSON: %s", e)
            return {
                "available": False,
                "triggered": False,
                "reason": f"Error parsing JSON: {e}",
            }

        critical_errors = int(data.get("critical_errors", 0))
        last_error = data.get("last_error")
        last_ts = data.get("last_timestamp")

        triggered = False
        reason = "Error checks OK."

        if critical_errors >= self.cfg.max_critical_errors:
            triggered = True
            reason = (
                f"Critical error threshold exceeded: {critical_errors} ≥ "
                f"{self.cfg.max_critical_errors}."
            )

        return {
            "available": True,
            "triggered": triggered,
            "reason": reason,
            "critical_errors": critical_errors,
            "last_error": last_error,
            "last_timestamp": last_ts,
            "thresholds": {
                "max_critical_errors": self.cfg.max_critical_errors,
            },
        }
