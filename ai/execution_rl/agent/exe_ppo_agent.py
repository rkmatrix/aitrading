"""
ai/execution_rl/agent/exe_ppo_agent.py
------------------------------------------------------------
ExecutionRLAgent – adaptive execution brain (Phase 26 + Phase 60).
• Async batch routing via SmartOrderRouter
• HOLD intents skipped
• Online fine-tuning bridge auto-loads PPO model
------------------------------------------------------------
"""
from __future__ import annotations
import asyncio
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from ai.execution.smart_order_router import SmartOrderRouter

# Phase 60 Online Fine-Tuning
try:
    from ai.training.online_finetune import OnlineFineTuneBridge
except Exception:
    OnlineFineTuneBridge = None

logger = logging.getLogger("ExecutionRLAgent")


class ExecutionRLAgent:
    def __init__(self, model=None, router: SmartOrderRouter | None = None,
                 symbols=None, config=None):

        self.router = router
        self.symbols = symbols or ["AAPL", "MSFT", "TSLA"]
        self.config = config or {}

        self.exploration = self.config.get("exploration", 0.05)
        self.max_concurrent = self.config.get("max_concurrent", 3)

        # PHASE 60 MODEL LOAD / ONLINE FINE TUNE
        self.model = None
        if OnlineFineTuneBridge:
            try:
                bridge = OnlineFineTuneBridge("configs/phase60_online_finetune.yaml")
                self.model = bridge.tuner.model
            except Exception as e:
                logger.warning(f"⚠️ Phase60 disabled: {e}")

        # If still None – agent falls back to exploration-only mode
        logger.info(f"🧠 ExecutionRLAgent initialized (async-ready) for {len(self.symbols)} symbols.")

    async def execute_batch_async(self, market_obs: dict):
        tasks = []
        sem = asyncio.Semaphore(self.max_concurrent)

        async def safe_execute(symbol):
            async with sem:
                intent = self.decide_action(symbol, market_obs[symbol])
                order = self.intent_to_order(symbol, intent)

                if not order:
                    logger.debug(f"⏭️  {symbol}: HOLD (no order)")
                    return

                try:
                    result = await self.router.route_order_async(order)
                    logger.info(f"✅ Async executed {symbol}: {result}")
                    self.update_reward(symbol, intent, result)
                except Exception as e:
                    logger.error(f"💥 Async routing failed for {symbol}: {e}")

        for sym in self.symbols:
            tasks.append(asyncio.create_task(safe_execute(sym)))

        await asyncio.gather(*tasks)

    def decide_action(self, symbol: str, obs) -> dict:
        if np.random.rand() < self.exploration:
            action = np.random.choice(["BUY", "SELL", "HOLD"])
        else:
            if self.model and hasattr(self.model, "predict"):
                action, _ = self.model.predict(obs, deterministic=True)
                action = str(action).upper()
            else:
                action = np.random.choice(["BUY", "SELL", "HOLD"])

        logger.debug(f"🤖 {symbol} → RL decided: {action}")
        return {"symbol": symbol, "action": action, "confidence": 1.0}

    def intent_to_order(self, symbol: str, intent: dict) -> dict | None:
        side = intent.get("action", "HOLD").upper()
        if side == "HOLD":
            return None

        qty = int(self.config.get("default_qty", 10))
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": "MARKET",
            "limit_price": None,
            "tag": f"rl_{datetime.utcnow().strftime('%H%M%S')}",
        }

    def update_reward(self, symbol: str, intent: dict, result: dict):
        if not result:
            return
        fill_price = result.get("fill_price")
        side = intent.get("action")
        logger.info(f"🏁 Reward update: {symbol} {side} @ {fill_price}")

    async def run_once(self, obs_batch: dict):
        logger.info("🚀 Phase26 RL Agent async execution cycle started.")
        await self.execute_batch_async(obs_batch)
        logger.info("✅ Async execution cycle complete.")
