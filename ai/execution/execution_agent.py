"""
ai/execution/execution_agent.py
Execution Agent module for AITradeBot.

Implements live reload via Phase 41.2 mixin to update execution
rules (e.g., slippage, timeouts, broker settings) dynamically.
"""

from __future__ import annotations
import logging, time, random

# ✅ Live reload mixin
from ai.execution.live_reload_mixin import LiveReloadMixin

logger = logging.getLogger(__name__)


class ExecutionAgent(LiveReloadMixin):
    """
    🤖 ExecutionAgent — handles actual trade placement and monitoring.

    This class respects dynamic exec policies, e.g. "execution_base".
    """

    def __init__(self, mode: str = "PAPER"):
        self.mode = mode
        self.last_exec_result = None
        logger.info("🤖 ExecutionAgent initialized in %s mode", mode)

    # ------------------------------------------------------------------
    # Trade execution logic
    # ------------------------------------------------------------------
    def execute_trade(self, symbol: str, qty: float, price: float) -> bool:
        """
        Simulate execution under an exec policy (slippage, latency, etc.)
        """
        policy = self.get_exec_policy("execution_base")
        slippage_bps = 10
        delay_ms = 100

        if policy:
            payload = policy.get("payload", {})
            slippage_bps = payload.get("slippage_bps", slippage_bps)
            delay_ms = payload.get("delay_ms", delay_ms)

        effective_price = price * (1 + slippage_bps / 10000.0)
        time.sleep(delay_ms / 1000.0)

        success = random.random() > 0.1
        self.last_exec_result = {
            "symbol": symbol,
            "qty": qty,
            "price": effective_price,
            "success": success,
        }

        if success:
            logger.info(
                "💥 Executed %s qty=%d @ %.2f (slippage %.1fbps)",
                symbol, qty, effective_price, slippage_bps
            )
        else:
            logger.warning("🚫 Execution failed for %s", symbol)

        return success

    # ------------------------------------------------------------------
    # Monitoring & diagnostics
    # ------------------------------------------------------------------
    def monitor(self):
        logger.debug("Monitoring execution health (mode=%s)", self.mode)


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

    parser = argparse.ArgumentParser(description="ExecutionAgent standalone test")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--qty", type=int, default=10)
    parser.add_argument("--price", type=float, default=150.0)
    args = parser.parse_args()

    agent = ExecutionAgent()

    # Load live policy registry if exists
    try:
        reg = PolicyRegistry("data/runtime/policy_registry.json").all()
        ExecutionAgent.reload_policies(reg)
    except Exception as ex:
        logger.warning("Policy registry load failed: %s", ex)

    result = agent.execute_trade(args.symbol, args.qty, args.price)
    print(f"✅ Trade result: {result}  details={agent.last_exec_result}")
