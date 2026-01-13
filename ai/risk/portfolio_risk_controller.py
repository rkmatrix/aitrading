"""
Portfolio Risk Controller (Phase 31)
Monitors exposure, volatility, and drawdowns — applies scaling or halts trades.
"""

from __future__ import annotations
import logging, time, numpy as np
from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)

class PortfolioRiskController:
    def __init__(self, cfg: dict, portfolio_brain):
        self.cfg = cfg
        self.brain = portfolio_brain
        self.last_check = 0
        self.interval = cfg.get("risk_limits", {}).get("rebalance_interval", 300)
        self.limits = cfg["risk_limits"]

        logger.info("🧠 PortfolioRiskController initialized with limits: %s", self.limits)

    # -----------------------------------------------------------
    def check_risk(self, force=False):
        """Main risk evaluation loop, returns dict of flags + suggestions."""
        now = time.time()
        if not force and (now - self.last_check) < self.interval:
            return {"status": "SKIPPED", "reason": "interval_not_elapsed"}

        self.last_check = now
        exposures = self.brain.get_current_exposures()
        total_exp = sum(exposures.values())

        breaches = []
        if total_exp > self.limits["max_total_exposure"]:
            breaches.append(f"Total exposure {total_exp:.2f} > {self.limits['max_total_exposure']}")

        for sym, val in exposures.items():
            if val > self.limits["max_symbol_exposure"]:
                breaches.append(f"{sym} exposure {val:.2f} > {self.limits['max_symbol_exposure']}")

        dd = self.brain.get_drawdown()
        if dd > self.limits["max_drawdown"]:
            breaches.append(f"Drawdown {dd:.2f} > {self.limits['max_drawdown']}")

        if breaches:
            msg = "⚠️ RISK BREACH DETECTED:\n" + "\n".join(breaches)
            logger.warning(msg)
            if self.cfg["actions"].get("telegram_alerts"):
                notify(msg, kind="guardian")

            if self.cfg["actions"].get("halt_on_breach"):
                self.brain.freeze_trading()
                return {"status": "HALTED", "breaches": breaches}

            if self.cfg["actions"].get("auto_reduce_positions"):
                self.brain.scale_positions(0.8)
                return {"status": "REDUCED", "breaches": breaches}

        logger.info("✅ Risk check passed (total=%.2f, drawdown=%.2f)", total_exp, dd)
        return {"status": "OK", "exposures": exposures, "drawdown": dd}
