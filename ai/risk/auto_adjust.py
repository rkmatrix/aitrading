from __future__ import annotations
from pathlib import Path
import json, yaml
from typing import Dict, Any, List

from .policies import BaseLimits, derive_limits, guardian_mode_for_grade, router_weights_for_grade

def _safe_read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def compute_adjustments(cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    eval_json = Path(cfg["eval_json"])
    base_cfg = cfg["base_limits"]
    policy_cfg = cfg["policy"]
    guardian_cfg = cfg.get("guardian", {})
    router_cfg = cfg.get("router", {})
    symbols: List[str] = cfg.get("symbols", ["AAPL", "MSFT", "TSLA"])

    res = _safe_read_json(eval_json)
    grade = res.get("grade", "N/A")
    metrics = res.get("metrics", {}) or {}
    max_dd = float(metrics.get("max_drawdown", 0.0))

    base = BaseLimits(
        max_position_pct=float(base_cfg["max_position_pct"]),
        gross_leverage=float(base_cfg["gross_leverage"]),
        order_rate_per_min=int(base_cfg["order_rate_per_min"]),
    )
    limits = derive_limits(grade, max_dd, base, policy_cfg)
    guardian_mode = guardian_mode_for_grade(grade, guardian_cfg)
    router_w = router_weights_for_grade(grade, router_cfg)

    per_symbol_caps = {s: limits.max_position_pct for s in symbols}

    adjustments = {
        "grade": grade,
        "max_drawdown": max_dd,
        "limits": {
            "max_position_pct": limits.max_position_pct,
            "gross_leverage": limits.gross_leverage,
            "order_rate_per_min": limits.order_rate_per_min,
        },
        "guardian": {"mode": guardian_mode},
        "router": {"weights": router_w},
        "per_symbol_caps": per_symbol_caps,
        "notes": limits.notes,
        "source": str(eval_json),
    }
    return adjustments

def write_adjustments(adjustments: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(adjustments, indent=2), encoding="utf-8")
