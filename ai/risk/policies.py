from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BaseLimits:
    max_position_pct: float
    gross_leverage: float
    order_rate_per_min: int

@dataclass
class AdjustedLimits:
    max_position_pct: float
    gross_leverage: float
    order_rate_per_min: int
    grade: str
    max_drawdown: float
    notes: str

def _clamp(val: float, mn: float, mx: float) -> float:
    return max(mn, min(mx, val))

def derive_limits(grade: str, max_dd: float, base: BaseLimits, policy_cfg: Dict[str, Any]) -> AdjustedLimits:
    scale_map: Dict[str, Any] = policy_cfg["scale_map"]
    clamps: Dict[str, Any] = policy_cfg["clamps"]
    dd_overrides = policy_cfg.get("dd_overrides", [])

    delta = scale_map.get(grade, {"position_pct_delta": 0.0, "leverage_delta": 0.0, "throttle_delta": 0})
    # apply grade deltas
    new_pos = base.max_position_pct + float(delta.get("position_pct_delta", 0.0))
    new_lev = base.gross_leverage + float(delta.get("leverage_delta", 0.0))
    new_thr = base.order_rate_per_min + int(delta.get("throttle_delta", 0))

    # clamp to safety bounds
    new_pos = _clamp(new_pos, clamps["max_position_pct_min"], clamps["max_position_pct_max"])
    new_lev = _clamp(new_lev, clamps["gross_leverage_min"], clamps["gross_leverage_max"])
    new_thr = int(_clamp(new_thr, clamps["order_rate_min"], clamps["order_rate_max"]))

    notes = [f"grade={grade} deltas applied"]

    # drawdown overrides (strongest rule where dd <= threshold)
    enforced = {}
    for rule in sorted(dd_overrides, key=lambda r: r["dd_lte"]):  # ascending thresholds
        if max_dd <= float(rule["dd_lte"]):
            enforced = rule
    if enforced:
        if "max_position_pct_cap" in enforced:
            new_pos = min(new_pos, float(enforced["max_position_pct_cap"]))
        if "gross_leverage_cap" in enforced:
            new_lev = min(new_lev, float(enforced["gross_leverage_cap"]))
        if "order_rate_cap" in enforced:
            new_thr = min(new_thr, int(enforced["order_rate_cap"]))
        notes.append(f"drawdown override applied (dd={max_dd:.3f})")

    return AdjustedLimits(
        max_position_pct=new_pos,
        gross_leverage=new_lev,
        order_rate_per_min=new_thr,
        grade=grade,
        max_drawdown=max_dd,
        notes="; ".join(notes),
    )

def guardian_mode_for_grade(grade: str, guardian_cfg: Dict[str, Any]) -> str:
    mode_map = guardian_cfg.get("mode_map", {})
    return mode_map.get(grade, "standard")


def router_weights_for_grade(grade: str, router_cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns suggested (primary, backup) target weights.
    Handles unknown or N/A grades safely.
    """
    prefer_grade = router_cfg.get("prefer_primary_if_grade_at_least", "B+")
    backup_weight_on_bad = float(router_cfg.get("backup_weight_on_bad_grades", 0.65))

    # Fallback for missing or invalid grade
    valid_grades = ["E", "D", "C", "B", "B+", "A-", "A", "A+"]
    if grade not in valid_grades:
        return {"primary": 0.5, "backup": 0.5}

    # Compare grades on predefined scale
    order = valid_grades
    if order.index(grade) < order.index(prefer_grade):
        # poor grade → bias backup (safer route)
        return {"primary": 1.0 - backup_weight_on_bad, "backup": backup_weight_on_bad}

    # good grade → favor primary broker
    return {"primary": 0.8, "backup": 0.2}

