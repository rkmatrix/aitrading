from __future__ import annotations
from typing import Dict, Any, List

class SelfCritiqueLayer:
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", True))
        self.pens = cfg.get("penalties", {})
        self.bons = cfg.get("bonuses", {})
        self.thr = cfg.get("thresholds", {})
        self.regime_cfg = cfg.get("regime", {})
        self._trade_count_window: List[int] = []  # store per-step counts within hour

    def reset(self):
        self._trade_count_window.clear()

    def step(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"adj": 0.0, "reasons": []}

        adj = 0.0
        reasons: List[str] = []

        # 1) Stop-loss breach
        if info.get("breach_stop_loss", False):
            adj += float(self.pens.get("breach_stop_loss", 0.0))
            reasons.append("breach_stop_loss")

        # 2) Risk budget
        if info.get("exceed_risk_budget", False):
            adj += float(self.pens.get("exceed_risk_budget", 0.0))
            reasons.append("exceed_risk_budget")

        # 3) Overtrading
        trades_this_step = int(info.get("trades_this_step", 0))
        self._trade_count_window.append(trades_this_step)
        if len(self._trade_count_window) > 60:  # approx “hour” in 1-min bars
            self._trade_count_window.pop(0)
        max_trades_per_hour = int(self.thr.get("max_trades_per_hour", 12))
        if sum(self._trade_count_window) > max_trades_per_hour:
            adj += float(self.pens.get("overtrading", 0.0))
            reasons.append("overtrading")

        # 4) Slippage spike
        slip_bps = float(info.get("step_slippage_bps", 0.0))
        if abs(slip_bps) >= float(self.thr.get("slippage_spike_bps", 8)):
            adj += float(self.pens.get("slippage_spike", 0.0))
            reasons.append("slippage_spike")

        # 5) Regime mismatch (if env provides regime)
        k = self.regime_cfg.get("feature_key", "vol_regime")
        allowed = set(self.regime_cfg.get("allowed", []))
        if k in info and allowed:
            if str(info[k]) not in allowed:
                adj += float(self.pens.get("regime_mismatch", 0.0))
                reasons.append(f"regime_mismatch:{info[k]}")

        # 6) Positive reinforcement if stayed within plan
        if info.get("trade_within_plan", False):
            adj += float(self.bons.get("trade_within_plan", 0.0))
            reasons.append("trade_within_plan")

        return {"adj": float(adj), "reasons": reasons}
