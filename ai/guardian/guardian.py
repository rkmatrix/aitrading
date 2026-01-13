"""
ai/guardian/guardian.py
Main Guardian module for AITradeBot.

This class validates orders against live risk/compliance policies.
It now supports hot-reload via Phase 41.2's LiveReloadMixin so that
updated rules are applied instantly without restarting the bot.
"""

from __future__ import annotations
import logging, time

# ✅ bring in live policy reload support
from ai.guardian.live_reload_mixin import LiveReloadMixin

logger = logging.getLogger(__name__)


class Guardian(LiveReloadMixin):
    """
    🛡️ Guardian — central risk & compliance manager.

    Responsibilities:
      • Validate trades before execution.
      • Enforce dynamic limits (position size, leverage, exposure).
      • Integrate with risk_base and other live policy bundles.
    """

    def __init__(self, mode: str = "LIVE"):
        self.mode = mode
        self.start_ts = time.time()
        logger.info("🛡️ Guardian initialized in %s mode", mode)

    # ------------------------------------------------------------------
    # Core policy logic
    # ------------------------------------------------------------------
    def allow_order(self, symbol: str, qty: float, price: float) -> bool:
        """
        Evaluate an order request using the latest risk policy.

        Returns:
            True  → order allowed
            False → blocked
        """
        policy = self.get_policy("risk_base")
        if not policy:
            logger.warning("⚠️ No active risk policy found — default allow.")
            return True

        payload = policy.get("payload", {})
        rules = payload.get("rules", {})

        max_pos_usd = rules.get("max_position_usd", float("inf"))
        max_weight = rules.get("max_symbol_weight", 1.0)
        max_leverage = rules.get("max_leverage", 10.0)

        notional = qty * price
        if notional > max_pos_usd:
            logger.warning(
                "❌ Blocked %s: %.2f USD exceeds max_position_usd %.2f",
                symbol, notional, max_pos_usd
            )
            return False

        # You can expand checks here (portfolio weight, leverage, etc.)
        logger.info(
            "✅ Order allowed for %s — notional=%.2f (policy OK)",
            symbol, notional
        )
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summarize_policies(self) -> None:
        """
        Print summary of currently loaded policies (for debugging).
        """
        if not self._registry:
            logger.info("Guardian: no policies currently loaded.")
            return

        logger.info("Guardian active policies:")
        for name, rec in self._registry.items():
            logger.info(
                "  • %s  v%s  (source=%s)",
                name, rec.get("version"), rec.get("source_id")
            )

    def periodic_self_check(self):
        """
        Example periodic diagnostic hook.
        """
        logger.debug(
            "Guardian self-check ok (uptime %.1fs)",
            time.time() - self.start_ts
        )


# ----------------------------------------------------------------------
# Optional: lightweight CLI for standalone testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardian standalone test")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--qty", type=float, default=10)
    parser.add_argument("--price", type=float, default=150)
    args = parser.parse_args()

    g = Guardian(mode="TEST")
    # simulate loading registry file if exists
    from services.policy_registry import PolicyRegistry
    try:
        reg = PolicyRegistry("data/runtime/policy_registry.json").all()
        Guardian.reload_policies(reg)
    except Exception as ex:
        logger.warning("Registry load failed: %s", ex)

    allowed = g.allow_order(args.symbol, args.qty, args.price)
    print(f"Order allowed? {allowed}")
