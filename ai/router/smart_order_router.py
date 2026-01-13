"""
ai/router/smart_order_router.py
Smart Order Router module for AITradeBot.

Uses Phase 41.2 LiveReloadMixin to reload routing policies dynamically.
"""

from __future__ import annotations
import logging, random, time

# ✅ Live reload mixin
from ai.router.live_reload_mixin import LiveReloadMixin

logger = logging.getLogger(__name__)


class SmartOrderRouter(LiveReloadMixin):
    """
    📡 SmartOrderRouter — decides which broker or venue
    should handle an order based on routing policies.

    Reads router_equities or similar policy from registry.
    """

    def __init__(self, brokers: list[str] | None = None):
        self.brokers = brokers or ["alpaca", "backup"]
        self.last_route = None
        logger.info("🚀 SmartOrderRouter initialized with brokers=%s", self.brokers)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def route_order(self, symbol: str) -> str:
        """
        Decide which broker to route a symbol to.
        """
        policy = self.get_routing_policy("router_equities")
        if policy:
            routes = policy.get("payload", {}).get("routes", [])
            for entry in routes:
                if entry["symbol"].upper() == symbol.upper():
                    chosen = entry.get("primary", self.brokers[0])
                    self.last_route = chosen
                    logger.info("📡 Routed %s → %s (via policy)", symbol, chosen)
                    return chosen

        # fallback if no policy entry
        route = random.choice(self.brokers)
        self.last_route = route
        logger.warning("⚠️ No policy match for %s — fallback to %s", symbol, route)
        return route

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def refresh_brokers(self) -> None:
        logger.debug("Refreshing broker connections...")
        time.sleep(0.1)
        logger.debug("Broker pool refreshed.")


# ----------------------------------------------------------------------
# CLI for standalone testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from services.policy_registry import PolicyRegistry

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="SmartOrderRouter standalone test")
    parser.add_argument("--symbol", default="AAPL")
    args = parser.parse_args()

    router = SmartOrderRouter()

    # Load registry if available
    try:
        reg = PolicyRegistry("data/runtime/policy_registry.json").all()
        SmartOrderRouter.reload_policies(reg)
    except Exception as ex:
        logger.warning("Policy registry load failed: %s", ex)

    route = router.route_order(args.symbol)
    print(f"✅ Routed {args.symbol} via {route}")
