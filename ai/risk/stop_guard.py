import time
import logging

logger = logging.getLogger(__name__)

class StopGuard:
    """
    Enhanced Stop-Guard with detailed logging & recovery logic.
    - Triggers when cumulative PnL < -max_drawdown
    - Resets when PnL recovers above half that drawdown
    - Logs countdown during cooldown window
    """

    def __init__(self, cfg):
        risk_cfg = cfg.get("risk_aware", {})
        self.max_drawdown = float(risk_cfg.get("max_daily_drawdown", 0.05))
        self.cooldown = int(risk_cfg.get("cooldown_period", 300))
        self.last_trigger_time = None
        self.state = {"active": False, "reason": None, "pnl_at_trigger": 0.0}
        self.last_log_time = 0.0
        logger.info(
            "🛡️ StopGuard initialized: max_dd=%.2f%% cooldown=%ds",
            self.max_drawdown * 100, self.cooldown
        )

    # ------------------------------------------------------------------
    def update(self, info):
        """Check current PnL and trigger or reset guard as needed."""
        pnl = float(info.get("pnl", 0.0))

        # ✅ Auto-reset when PnL recovers
        if self.state["active"] and pnl > -abs(self.max_drawdown) * 0.5:
            self.state.update({"active": False, "reason": None})
            logger.info(
                "🟢 Stop-Guard reset — PnL recovered to %.3f (threshold %.3f).",
                pnl, -abs(self.max_drawdown) * 0.5
            )

        # 💥 Trigger on drawdown breach
        elif not self.state["active"] and pnl < -abs(self.max_drawdown):
            self.trigger("drawdown_limit", pnl)

        # ⏱️ Periodic cooldown countdown logging
        elif self.state["active"]:
            self._maybe_log_countdown()

    # ------------------------------------------------------------------
    def trigger(self, reason: str, pnl: float):
        """Activate Stop-Guard immediately."""
        self.state.update({"active": True, "reason": reason, "pnl_at_trigger": pnl})
        self.last_trigger_time = time.time()
        logger.error(
            "🚨 Stop-Guard TRIGGERED (%s): PnL=%.3f below %.3f (%.1f%% drawdown). "
            "Trading paused for %ds.",
            reason, pnl, -abs(self.max_drawdown), abs(pnl) * 100, self.cooldown
        )

    # ------------------------------------------------------------------
    def is_triggered(self) -> bool:
        """Return True if trading should be blocked."""
        if not self.state["active"]:
            return False

        elapsed = time.time() - self.last_trigger_time
        if elapsed > self.cooldown:
            logger.info("✅ Stop-Guard cooldown expired (%.1fs), re-enabling trades.", elapsed)
            self.state.update({"active": False, "reason": None})
            return False
        return True

    # ------------------------------------------------------------------
    def _maybe_log_countdown(self):
        """Emit countdown log every ~5 seconds while guard is active."""
        now = time.time()
        if now - self.last_log_time < 5:
            return
        self.last_log_time = now
        remaining = max(0, self.cooldown - int(now - self.last_trigger_time))
        logger.warning(
            "⏳ Stop-Guard active (%s). %.1f s remaining before re-enable. "
            "Triggered PnL=%.3f, threshold=%.3f.",
            self.state["reason"], remaining,
            self.state["pnl_at_trigger"], -abs(self.max_drawdown)
        )
