# ai/testing/failure_harness.py

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml
from unittest import mock

from ai.testing.failure_scenarios import FailureScenario, PatchSpec, build_scenarios

log = logging.getLogger("FailureHarness")


def _resolve_callable(module_name: str, callable_name: str) -> Callable[..., Any]:
    """
    Import and resolve the target callable.
    Example:
        module_name="runner.phase26_realtime_live"
        callable_name="main"
    """
    module = importlib.import_module(module_name)
    target = getattr(module, callable_name, None)
    if target is None:
        raise AttributeError(f"Callable '{callable_name}' not found in module '{module_name}'")
    if not callable(target):
        raise TypeError(f"Resolved attribute '{callable_name}' in '{module_name}' is not callable")
    return target


def _resolve_exception(exc_name: str) -> type[BaseException]:
    """
    Given a string like 'RuntimeError' or 'ValueError', returns
    the exception class, defaulting to RuntimeError if unknown.
    """
    # Try builtins first
    builtins_exc = getattr(__builtins__, exc_name, None)
    if isinstance(builtins_exc, type) and issubclass(builtins_exc, BaseException):
        return builtins_exc
    # Fallback generic
    return RuntimeError


def _resolve_original_callable(target: str) -> Optional[Callable[..., Any]]:
    """
    Given a dotted target string like:

        "ai.execution.market_feed.get_last_price"
        "ai.utils.alpaca_client.AlpacaClientAdapter.place_order"

    try to import the deepest valid module and then walk attributes
    to get the original callable.
    """
    parts = target.split(".")

    # Try all possible splits from right to left:
    # ai.utils.alpaca_client.AlpacaClientAdapter.place_order
    #  -> module: ai.utils.alpaca_client
    #     attrs:  AlpacaClientAdapter.place_order
    for split_idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_idx])
        attr_parts = parts[split_idx:]

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        obj: Any = module
        try:
            for attr in attr_parts:
                obj = getattr(obj, attr)
        except AttributeError:
            continue

        if callable(obj):
            return obj

    log.warning("Unable to resolve original callable for target %r", target)
    return None


@contextmanager
def _apply_patches(patches: Iterable[PatchSpec]) -> Iterable[mock._patch]:
    """
    Context manager to apply a collection of monkeypatches for the duration of a scenario.

    Supports both:
      - module-level functions: ai.execution.market_feed.get_last_price
      - class methods: ai.utils.alpaca_client.AlpacaClientAdapter.place_order
    """
    active_patches: List[mock._patch] = []

    def _make_side_effect(p: PatchSpec, original: Optional[Callable[..., Any]]) -> Callable[..., Any]:
        mode = p.mode.lower()

        if mode == "raise":
            exc_cls = _resolve_exception(p.exception or "RuntimeError")
            msg = p.message or f"Simulated failure for {p.target}"

            def _raise(*args, **kwargs):
                raise exc_cls(msg)

            return _raise

        if mode == "delay":
            delay_sec = float(p.delay_seconds or 1.0)

            def _delay(*args, **kwargs):
                time.sleep(delay_sec)
                if original is not None:
                    return original(*args, **kwargs)
                return None

            return _delay

        if mode == "return":
            val = p.return_value

            def _return(*args, **kwargs):
                return val

            return _return

        if mode == "noop":
            def _noop(*args, **kwargs):
                return None

            return _noop

        # Fallback: no-op wrapper around original
        def _fallback(*args, **kwargs):
            if original is not None:
                return original(*args, **kwargs)
            return None

        return _fallback

    try:
        for p in patches:
            if not p.enabled:
                continue

            original = _resolve_original_callable(p.target)
            side_effect = _make_side_effect(p, original)

            try:
                # IMPORTANT: use full dotted target here, so class methods work
                patcher = mock.patch(p.target, side_effect)
                patcher.start()
                active_patches.append(patcher)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to apply patch for %r: %s", p.target, e)

        yield active_patches

    finally:
        for patcher in active_patches:
            try:
                patcher.stop()
            except Exception:
                pass



class FailureModeTestHarness:
    """
    Core harness that:
      - Loads failure scenarios from YAML.
      - Runs target callable under each scenario.
      - Captures outcome, exceptions, and runtime.
      - Writes JSONL + CSV reports.
    """

    def __init__(
        self,
        *,
        config_path: str = "configs/phase122_failure_harness.yaml",
        report_dir: str = "data/reports",
    ) -> None:
        self.config_path = config_path
        self.report_dir = Path(report_dir).resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        fh_cfg = self.cfg.get("failure_harness", {})
        target_cfg = fh_cfg.get("target", {})

        self.target_module: str = target_cfg.get("module", "")
        self.target_callable_name: str = target_cfg.get("callable", "main")
        self.target_args: List[Any] = target_cfg.get("args", [])
        self.target_kwargs: Dict[str, Any] = target_cfg.get("kwargs", {})

        if not self.target_module:
            raise ValueError("failure_harness.target.module is required in config")

        self.scenarios: List[FailureScenario] = build_scenarios(self.cfg)

        self.jsonl_path = self.report_dir / "phase122_failures.jsonl"
        self.csv_path = self.report_dir / "phase122_failures.csv"

    def _run_once(
        self,
        target_fn: Callable[..., Any],
        scenario: FailureScenario,
    ) -> Dict[str, Any]:
        """
        Run the target callable under one failure scenario (single repetition).
        """
        start_ts = time.time()
        status = "success"
        exception_type: Optional[str] = None
        exception_msg: Optional[str] = None

        log.info("▶ Running scenario '%s' with %d patches", scenario.name, len(scenario.patches))

        with _apply_patches(scenario.patches):
            try:
                if scenario.max_runtime_seconds:
                    # Soft timeout via elapsed check in a wrapper
                    # (We do not kill threads/processes; just measure)
                    result = target_fn(*self.target_args, **self.target_kwargs)
                else:
                    result = target_fn(*self.target_args, **self.target_kwargs)

            except Exception as e:  # noqa: BLE001
                status = "exception"
                exception_type = e.__class__.__name__
                exception_msg = str(e)
                result = None

        elapsed = time.time() - start_ts

        record: Dict[str, Any] = {
            "scenario_name": scenario.name,
            "status": status,
            "exception_type": exception_type,
            "exception_message": exception_msg,
            "elapsed_seconds": elapsed,
            "tags": list(scenario.tags),
            "max_runtime_seconds": scenario.max_runtime_seconds,
            "result_repr": repr(result)[:500],
            "timestamp": start_ts,
        }

        # Include raw scenario definition (without patch objects)
        record["scenario"] = {
            "name": scenario.name,
            "description": scenario.description,
            "tags": scenario.tags,
            "max_runtime_seconds": scenario.max_runtime_seconds,
            "repeat": scenario.repeat,
        }

        return record

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run all scenarios, writing JSONL + CSV.

        Returns (records, summary).
        """
        target_fn = _resolve_callable(self.target_module, self.target_callable_name)
        all_records: List[Dict[str, Any]] = []

        for scenario in self.scenarios:
            for i in range(scenario.repeat or 1):
                log.info("=== Scenario '%s' run %d/%d ===", scenario.name, i + 1, scenario.repeat or 1)
                rec = self._run_once(target_fn, scenario)
                rec["scenario_run_index"] = i + 1
                all_records.append(rec)
                self._append_jsonl(rec)

        self._write_csv(all_records)
        summary = self._summarize(all_records)
        self._write_summary(summary)

        log.info("✅ Phase 122 Failure Harness completed: %d runs, %d scenarios", len(all_records), len(self.scenarios))
        return all_records, summary

    # ---------- Reporting helpers ----------

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_csv(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return

        import csv  # local import to avoid unused when not used

        # Flatten keys
        fieldnames = sorted(
            {
                k
                for rec in records
                for k in rec.keys()
                if k not in {"scenario"}  # scenario is nested; skip here
            }
        )

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = {k: v for k, v in rec.items() if k in fieldnames}
                writer.writerow(row)

    def _summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_runs": len(records),
            "by_scenario": {},
        }
        for rec in records:
            name = rec["scenario_name"]
            s_stat = summary["by_scenario"].setdefault(
                name,
                {"runs": 0, "success": 0, "exception": 0, "avg_elapsed": 0.0},
            )
            s_stat["runs"] += 1
            s_stat[rec["status"]] += 1

        # Compute averages
        for name, stats in summary["by_scenario"].items():
            elapsed_sum = sum(
                rec["elapsed_seconds"]
                for rec in records
                if rec["scenario_name"] == name
            )
            stats["avg_elapsed"] = elapsed_sum / max(stats["runs"], 1)

        return summary

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        summary_path = self.report_dir / "phase122_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
