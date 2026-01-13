"""
ai/guardian/risk_guardian.py
----------------------------
Central risk management and action validation layer for AITradeBot.

Responsibilities:
    • Validate every trade or RL action before routing.
    • Enforce position, drawdown, and cooldown limits.
    • Allow integration with guardian adapters in later phases.
    • Compatible with both LIVE and PAPER modes.

Used in:
    Phase 24 → Guarded Execution
    Phase 35 → SmartOrderRouter
    Phase 47 → Rollback Guardian
    Phase 50 → Intelligent Executor
"""

from __future__ import annotations
import os, json, logging, time
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ======================================================
# 🛡️ Guardian Base
# ======================================================
class Guardian:
    def __init__(self, mode: str = None, cfg: dict | None = None):
        self.mode = mode or os.getenv("MODE", "LIVE").upper()
        self.cfg = cfg or {}
        self.cooldown_seconds = self.cfg.get("cooldown_seconds", 30)
        self.max_position_value = self.cfg.get("max_position_value", 25000)
        self.max_drawdown = self.cfg.get("max_drawdown", 0.15)
        self.last_action_time: dict[str, float] = {}
        self.positions: dict[str, float] = {}  # simulated current holdings
        self.pnl_history: list[float] = []     # track recent PnL deltas
        logger.info(f"🛡️ Guardian initialized in {self.mode} mode")

    # --------------------------------------------------
    def record_pnl(self, pnl_value: float):
        """Optional: track realized PnL over time for drawdown monitoring."""
        self.pnl_history.append(pnl_value)
        if len(self.pnl_history) > 1000:
            self.pnl_history.pop(0)

    # --------------------------------------------------
    def _check_cooldown(self, symbol: str) -> bool:
        """Prevent over-trading a symbol too frequently."""
        now = time.time()
        last = self.last_action_time.get(symbol, 0)
        if now - last < self.cooldown_seconds:
            logger.info(f"⏳ Cooldown active for {symbol} ({now - last:.1f}s elapsed)")
            return False
        self.last_action_time[symbol] = now
        return True

    # --------------------------------------------------
    def _check_position_limit(self, symbol: str, action: str) -> bool:
        """Simulated position limit control."""
        pos = self.positions.get(symbol, 0)
        if action == "BUY" and pos >= self.max_position_value:
            logger.warning(f"🚫 Position limit exceeded for {symbol}: {pos} >= {self.max_position_value}")
            return False
        if action == "SELL" and pos <= -self.max_position_value:
            logger.warning(f"🚫 Short limit exceeded for {symbol}: {pos} <= -{self.max_position_value}")
            return False
        return True

    # --------------------------------------------------
    def _check_drawdown(self) -> bool:
        """Simple simulated drawdown check."""
        if not self.pnl_history:
            return True
        peak = max(self.pnl_history)
        trough = min(self.pnl_history)
        if peak <= 0:
            return True
        dd = (peak - trough) / max(1e-6, peak)
        if dd > self.max_drawdown:
            logger.warning(f"⚠️ Drawdown limit exceeded ({dd:.2%} > {self.max_drawdown:.2%})")
            return False
        return True

    # --------------------------------------------------
    def _check_risk(self, act: dict) -> bool:
        """Aggregate risk checks for a given action."""
        sym = act.get("symbol")
        action = act.get("action")
        if not self._check_cooldown(sym):
            return False
        if not self._check_position_limit(sym, action):
            return False
        if not self._check_drawdown():
            return False
        return True

    # --------------------------------------------------
    def allow_action(self, act: dict) -> bool:
        """Primary entry point for risk validation (legacy compatibility)."""
        return self._check_risk(act)

    # --------------------------------------------------
    # ✅ Universal adapter for IntelligentExecutor
    def is_action_allowed(self, act):
        """
        Adapter so any Guardian variant can integrate with Phase 50 executor.
        It should return True if action is permitted, False otherwise.
        """
        try:
            # If a custom internal method exists, delegate to it
            for cand in ["allow_action", "allow", "approve_trade", "check", "validate"]:
                if hasattr(self, cand):
                    fn = getattr(self, cand)
                    try:
                        return fn(act)
                    except TypeError:
                        # handle signature like (symbol, action)
                        return fn(act.get("symbol"), act.get("action"))
            # Fallback to internal _check_risk
            return self._check_risk(act)
        except Exception as e:
            logger.warning(f"Guardian adapter error: {e}")
            return True


# ======================================================
# 📦 Factory wrapper (for backward compatibility)
# ======================================================
def RiskGuardian(mode: str = None, cfg: dict | None = None) -> Guardian:
    """Alias factory so imports using RiskGuardian() still work."""
    return Guardian(mode=mode, cfg=cfg)
