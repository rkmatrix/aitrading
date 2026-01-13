"""
Dynamic Risk Adjustment Loop (Phase 32)
Continuously adapts risk limits from PortfolioRiskController based on
volatility and PnL feedback.
"""

from __future__ import annotations
import time, logging, numpy as np
from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)

class DynamicRiskLoop:
    def __init__(self, cfg: dict, risk_controller, portfolio_brain):
        self.cfg = cfg
        self.controller = risk_controller
        self.brain = portfolio_brain
        self.last_adj = 0
        self.params = cfg.get("adaptive_params", {})
        self.cool_down = self.params.get("cool_down", 300)

        logger.info("🧩 DynamicRiskLoop initialized with params: %s", self.params)

    def compute_volatility(self) -> float:
        """Stub for realized volatility — random demo value."""
        return abs(np.random.normal(0.2, 0.05))

    def compute_recent_pnl(self) -> float:
        """Compute recent average PnL (simulated)."""
        if not self.brain.trade_history:
            return 0.0
        costs = [abs(t["cost"]) for t in self.brain.trade_history[-self.params.get("pnl_window", 50):]]
        return float(np.mean(costs) / (np.std(costs) + 1e-9))

    def adjust_limits(self):
        now = time.time()
        if (now - self.last_adj) < self.cool_down:
            return {"status": "SKIPPED", "reason": "cool_down"}

        self.last_adj = now
        vol = self.compute_volatility()
        pnl = self.compute_recent_pnl()

        limits = self.controller.limits.copy()
        step = self.params.get("adjust_step", 0.05)

        # Simple logic: tighten when volatility high or PnL weak
        if vol > 0.3 or pnl < 0.5:
            limits["max_total_exposure"] = max(
                self.params["min_exposure"], limits["max_total_exposure"] - step
            )
            limits["max_drawdown"] = max(
                self.params["min_drawdown"], limits["max_drawdown"] - step
            )
            action = "TIGHTENED"
        else:
            limits["max_total_exposure"] = min(
                self.params["max_exposure"], limits["max_total_exposure"] + step
            )
            limits["max_drawdown"] = min(
                self.params["max_drawdown"], limits["max_drawdown"] + step
            )
            action = "LOOSENED"

        self.controller.limits.update(limits)
        msg = f"🔁 Risk limits {action}: total_exp={limits['max_total_exposure']:.2f}, drawdown={limits['max_drawdown']:.2f}"
        logger.info(msg)
        notify(msg, kind="guardian")
        return {"status": action, "volatility": vol, "pnl": pnl}
