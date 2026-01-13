import math
import operator
from typing import Dict, Any
from datetime import timedelta
from .schemas import Signal
from .feature_bus import FeatureBus
from ai.utils.time_utils import utcnow
from ai.utils.log_utils import get_logger

OPS = {
    ">":  operator.gt,
    "<":  operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}

class SignalFusionEngine:
    """
    Pulls recent features from the FeatureBus, applies rule-based logic,
    and emits a normalized Signal in real-time.
    """
    def __init__(self, cfg: Dict[str, Any], bus: FeatureBus):
        self.cfg = cfg
        self.bus = bus
        self.logger = get_logger("SignalFusion")
        self.required = [f["name"] for f in cfg["features"] if f.get("required")]
        self.feature_names = [f["name"] for f in cfg["features"]]
        self.max_stale_sec = cfg.get("max_feature_staleness_sec", 5)

    def _parse_clause(self, fval: float, expr: str) -> bool:
        expr = expr.strip()
        for op in (">=", "<=", "==", ">", "<"):
            if op in expr:
                rhs = float(expr.split(op)[1].strip())
                return OPS[op](fval, rhs)
        raise ValueError(f"Bad expression: {expr}")

    def _staleness_ok(self, ts) -> bool:
        return (utcnow() - ts) <= timedelta(seconds=self.max_stale_sec)

    def _scale_size(self, base_qty: int, momentum: float, max_scale: float) -> int:
        scale = 1.0 + min(max(momentum, -1.0), 1.0) * (max_scale - 1.0)
        return max(1, int(round(base_qty * scale)))

    def fuse(self, symbol: str) -> Signal | None:
        snap = self.bus.snapshot(symbol, self.feature_names)
        # Required present & fresh?
        for req in self.required:
            f = snap.get(req)
            if not f or not self._staleness_ok(f.ts):
                self.logger.debug("Missing/stale required feature %s for %s", req, symbol)
                return None

        strat = self.cfg["strategy"]
        vals = {k: (snap[k].value if snap.get(k) else float("nan")) for k in self.feature_names}

        def all_true(block: Dict[str,str]) -> bool:
            return all(not math.isnan(vals[k]) and self._parse_clause(vals[k], expr)
                       for k, expr in block.items())

        side = "FLAT"
        reason = "no-rule"
        strength = 0.0

        if "flat_if" in strat and all_true(strat["flat_if"]):
            side, reason, strength = "FLAT", "flat_if", 0.0
        elif "long_if" in strat and all_true(strat["long_if"]):
            side, reason, strength = "BUY", "long_if", 1.0
        elif "short_if" in strat and all_true(strat["short_if"]):
            side, reason, strength = "SELL", "short_if", -1.0

        size_cfg = strat.get("size", {})
        base_qty = int(size_cfg.get("base_qty", 1))
        if size_cfg.get("scale_with_momentum") and not math.isnan(vals.get("mom_1m", float("nan"))):
            size = self._scale_size(base_qty, abs(vals["mom_1m"]), float(size_cfg.get("max_scale", 2.0)))
        else:
            size = base_qty

        return Signal(symbol=symbol, side=side, strength=strength, size=size,
                      reason=reason, meta={"features": {k: v for k, v in vals.items()}})
