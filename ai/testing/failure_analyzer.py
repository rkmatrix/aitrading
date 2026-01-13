# ai/testing/failure_analyzer.py  (FIXED VERSION)

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger("FailureDiagnostic")


# -----------------------------------------------------
# CONFIG STRUCT
# -----------------------------------------------------
@dataclass
class DiagnosticConfig:
    perf_latency_threshold: float = 3.0
    min_runs_for_confidence: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "DiagnosticConfig":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            logger.warning("Diagnostic config not found: %s (using defaults)", path)
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        diag_cfg = cfg.get("diagnostic", {})
        return cls(
            perf_latency_threshold=float(diag_cfg.get("perf_latency_threshold", 3.0)),
            min_runs_for_confidence=int(diag_cfg.get("min_runs_for_confidence", 1)),
        )


# -----------------------------------------------------
# RECORD STRUCT
# -----------------------------------------------------
@dataclass
class FailureRunRecord:
    scenario_name: str
    status: str
    exception_type: Optional[str]
    exception_message: Optional[str]
    elapsed_seconds: float
    tags: List[str] = field(default_factory=list)
    scenario_run_index: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------
# LOAD RECORDS WITH FALLBACK & BETTER ERROR HANDLING
# -----------------------------------------------------
def load_failure_records(jsonl_path: str | Path) -> List[FailureRunRecord]:
    """
    Loads Phase 122 failure harness output.
    Includes fallback paths and clear errors.
    """
    jsonl_path = Path(jsonl_path).resolve()

    # MAIN CHECK
    if not jsonl_path.exists():
        logger.error("❌ JSONL file NOT found at exact path: %s", jsonl_path)

        # TRY FALLBACKS
        fallback1 = jsonl_path.parent / "phase122_failures.JSONL"
        fallback2 = jsonl_path.parent / "Phase122_failures.jsonl"

        for fb in (fallback1, fallback2):
            if fb.exists():
                logger.warning("⚠️ Using fallback JSONL found at: %s", fb)
                jsonl_path = fb
                break
        else:
            raise FileNotFoundError(
                f"FailureHarness JSONL not found anywhere.\n"
                f"Expected: {jsonl_path}\n"
                f"Make sure Phase 122 Harness was run."
            )

    logger.info("📄 Loading JSONL: %s", jsonl_path)
    records: List[FailureRunRecord] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            rec = FailureRunRecord(
                scenario_name=raw.get("scenario_name", ""),
                status=raw.get("status", "unknown"),
                exception_type=raw.get("exception_type"),
                exception_message=raw.get("exception_message"),
                elapsed_seconds=float(raw.get("elapsed_seconds", 0.0)),
                tags=list(raw.get("tags", [])),
                scenario_run_index=int(raw.get("scenario_run_index", 1)),
                raw=raw,
            )
            records.append(rec)

    if not records:
        raise ValueError(f"JSONL was loaded but contains **zero** records: {jsonl_path}")

    logger.info("📊 Loaded %d failure-run records", len(records))
    return records


# -----------------------------------------------------
# CLASSIFICATION LOGIC (unchanged)
# -----------------------------------------------------
def _classify_run(rec: FailureRunRecord, cfg: DiagnosticConfig) -> str:
    if rec.status == "exception":
        return "crash"
    if rec.status == "success" and not rec.exception_type:
        if any(tag in rec.tags for tag in ["latency", "chaos"]) and rec.elapsed_seconds > cfg.perf_latency_threshold:
            return "perf_degradation"
        if any(tag in rec.tags for tag in ["broker", "data", "risk", "policy", "portfolio"]):
            return "tolerated_failure"
        return "ok"
    return "unknown"


# -----------------------------------------------------
# AGGREGATION + REPORT GENERATION
# -----------------------------------------------------
@dataclass
class ScenarioDiagnostic:
    scenario_name: str
    tags: List[str]
    runs: int
    crashes: int
    perf_degradations: int
    tolerated_failures: int
    ok_runs: int
    worst_classification: str
    avg_elapsed: float
    max_elapsed: float
    example_exception: Optional[str] = None

    def severity_rank(self) -> int:
        mapping = {
            "CRITICAL_CRASH": 0,
            "WARN_PERF_DEGRADE": 1,
            "OK_HANDLED_FAILURE": 2,
            "OK": 3,
            "UNKNOWN": 4,
        }
        return mapping.get(self.worst_classification, 4)


def analyse_scenarios(records: Iterable[FailureRunRecord], cfg: DiagnosticConfig):
    by_scenario = {}
    for r in records:
        by_scenario.setdefault(r.scenario_name, []).append(r)

    results = []

    for scen, runs in by_scenario.items():
        tags = set()
        crashes = perf = tolerated = ok = 0
        exc_example = None
        elapsed_list = []

        for r in runs:
            tags.update(r.tags)
            elapsed_list.append(r.elapsed_seconds)

            label = _classify_run(r, cfg)

            if label == "crash":
                crashes += 1
                if not exc_example and r.exception_type:
                    exc_example = f"{r.exception_type}: {r.exception_message}"
            elif label == "perf_degradation":
                perf += 1
            elif label == "tolerated_failure":
                tolerated += 1
            elif label == "ok":
                ok += 1

        # Worst label
        if crashes:
            worst = "CRITICAL_CRASH"
        elif perf:
            worst = "WARN_PERF_DEGRADE"
        elif tolerated:
            worst = "OK_HANDLED_FAILURE"
        elif ok:
            worst = "OK"
        else:
            worst = "UNKNOWN"

        results.append(
            ScenarioDiagnostic(
                scenario_name=scen,
                tags=sorted(tags),
                runs=len(runs),
                crashes=crashes,
                perf_degradations=perf,
                tolerated_failures=tolerated,
                ok_runs=ok,
                worst_classification=worst,
                avg_elapsed=sum(elapsed_list) / len(elapsed_list),
                max_elapsed=max(elapsed_list),
                example_exception=exc_example,
            )
        )

    return sorted(results, key=lambda d: (d.severity_rank(), d.scenario_name))


# -----------------------------------------------------
# REPORT WRITERS
# -----------------------------------------------------
def build_markdown_report(diags):
    lines = ["# Phase 122.2 Failure Diagnostic Report", ""]
    if not diags:
        return "\n".join(lines + ["No diagnostics generated."])

    # Summary
    total = len(diags)
    critical = sum(1 for d in diags if d.worst_classification == "CRITICAL_CRASH")
    perf = sum(1 for d in diags if d.worst_classification == "WARN_PERF_DEGRADE")
    toler = sum(1 for d in diags if d.worst_classification == "OK_HANDLED_FAILURE")
    ok = sum(1 for d in diags if d.worst_classification == "OK")

    lines.append("## Summary")
    lines.append(f"- Total scenarios: **{total}**")
    lines.append(f"- Critical crashes: **{critical}**")
    lines.append(f"- Performance warnings: **{perf}**")
    lines.append(f"- Handled failures: **{toler}**")
    lines.append(f"- Passed: **{ok}**")
    lines.append("")

    # Details
    lines.append("## Details by Scenario\n")
    for d in diags:
        lines.append(f"### {d.scenario_name}")
        lines.append(f"- Classification: **{d.worst_classification}**")
        lines.append(f"- Tags: {', '.join(d.tags)}")
        lines.append(f"- Runs: {d.runs}")
        lines.append(f"- Crashes: {d.crashes} | Perf: {d.perf_degradations} | Handled: {d.tolerated_failures} | OK: {d.ok_runs}")
        lines.append(f"- Avg elapsed: `{d.avg_elapsed:.3f}s` (max `{d.max_elapsed:.3f}s`)")
        if d.example_exception:
            lines.append(f"- Example exception: `{d.example_exception}`")
        lines.append("")
    return "\n".join(lines)


def write_diagnostic_outputs(diags, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "phase122_diagnostic_report.json"
    md_path = report_dir / "phase122_diagnostic_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([d.__dict__ for d in diags], f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(diags))

    logger.info("Diagnostic reports written:")
    logger.info("  JSON → %s", json_path)
    logger.info("  MD   → %s", md_path)

    return {"json": str(json_path), "md": str(md_path)}
