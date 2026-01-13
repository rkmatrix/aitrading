from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import time

from .rules.base import RiskRule, RuleDecision
from .rules.max_position_rule import MaxPositionRule
from .rules.max_exposure_rule import MaxExposureRule
from .rules.drawdown_circuit_rule import DrawdownCircuitRule
from .rules.volatility_throttle_rule import VolatilityThrottleRule
from .rules.earnings_halt_rule import EarningsHaltRule
from .rules.news_halt_rule import NewsHaltRule

@dataclass
class Intent:
    symbol: str
    side: str
    qty: float
    order_type: str = "limit"
    limit_price: float | None = None
    ref_price: float | None = None

class RiskGuardian:
    def __init__(self, cfg: Dict[str, Any], flag_getter: Callable[[str, Any], Any] | None = None):
        self.cfg = cfg
        self.flag_getter = flag_getter or (lambda path, default=None: default)
        self.rules: List[RiskRule] = []
        rconf = cfg.get("risk", {}).get("rules", {})

        if rconf.get("max_position", {}).get("enabled", True):
            mpc = rconf["max_position"]
            self.rules.append(MaxPositionRule(mpc.get("max_pct_equity_per_symbol", 0.15),
                                              mpc.get("max_shares_per_symbol")))
        if rconf.get("max_exposure", {}).get("enabled", True):
            me = rconf["max_exposure"]
            self.rules.append(MaxExposureRule(me.get("long_gross_limit_pct", 150),
                                              me.get("short_gross_limit_pct", 50),
                                              me.get("net_limit_pct", 120)))
        if rconf.get("drawdown_circuit", {}).get("enabled", True):
            dd = rconf["drawdown_circuit"]
            self.rules.append(DrawdownCircuitRule(dd.get("intraday_dd_limit_pct", 5.0),
                                                  dd.get("daily_dd_limit_pct", 7.0),
                                                  dd.get("cooloff_minutes", 90)))
        if rconf.get("volatility_throttle", {}).get("enabled", True):
            vt = rconf["volatility_throttle"]
            self.rules.append(VolatilityThrottleRule(vt.get("vix_threshold", 30.0),
                                                     vt.get("reduce_size_factor", 0.5)))
        if rconf.get("earnings_halt", {}).get("enabled", True):
            eh = rconf["earnings_halt"]
            self.rules.append(EarningsHaltRule(eh.get("pre_earnings_halt_days", 2),
                                               eh.get("post_earnings_halt_days", 1)))
        if rconf.get("news_halt", {}).get("enabled", False):
            nh = rconf["news_halt"]
            self.rules.append(NewsHaltRule(nh.get("severity_threshold", "high")))

    def evaluate(self, intent: Intent, snapshot: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # Optional feature flag to disable all risk checks
        if self.flag_getter("risk.disable_all", False):
            return {"allow": True, "decisions": [], "mutations": {}}

        decisions: List[Dict[str, Any]] = []
        mutations: Dict[str, Any] = {}

        for rule in self.rules:
            d: RuleDecision = rule.check(intent, snapshot, ctx)
            decisions.append({"rule": rule.name, "allow": d.allow, "reason": d.reason, "details": d.details})
            # Merge mutations if present
            m = ctx.get("mutations")
            if m:
                for k, v in m.items():
                    mutations[k] = v

            if not d.allow:
                return {"allow": False, "decisions": decisions, "mutations": mutations}

        return {"allow": True, "decisions": decisions, "mutations": mutations}
