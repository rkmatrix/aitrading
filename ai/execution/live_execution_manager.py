from __future__ import annotations
import os
import time
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any

from ai.execution.execution_policy import ExecutionPolicy, ObsSpec
from ai.execution.execution_router import ExecutionRouter, RouterConfig
from utils.telegram_notifier import TelegramNotifier  # optional helper from earlier phases
from broker.broker_alpaca import BrokerAlpaca  # your existing broker wrapper


class LiveExecutionManager:
    """
    Bridges the RL execution policy with the Alpaca broker.
    - Loads PPO model
    - Routes incoming trade signals through the policy
    - Executes via BrokerAlpaca (live or dry-run)
    - Logs results to CSV for future training
    """

    def __init__(
        self,
        model_path: str,
        router_cfg: RouterConfig,
        dry_run: bool = True,
        log_path: str = "data/logs/live_exec/live_exec_log.csv",
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
    ):
        self.obs_spec = ObsSpec(inventory_limit=router_cfg.inventory_limit)
        self.policy = ExecutionPolicy.load(model_path, obs_spec=self.obs_spec)
        self.router = ExecutionRouter(policies={"GLOBAL": self.policy}, cfg=router_cfg)
        self.broker = BrokerAlpaca(dry_run=dry_run)
        self.notifier = (
            TelegramNotifier(telegram_token, telegram_chat_id)
            if telegram_token and telegram_chat_id
            else None
        )
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.dry_run = dry_run
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single trade signal through the RL policy + broker.
        Expected signal keys:
            symbol, side, qty, mid, spread
        """
        order = {
            "symbol": signal["symbol"],
            "side": signal["side"],
            "qty": float(signal["qty"]),
            "mid": float(signal["mid"]),
            "spread": float(signal.get("spread", 0.01)),
            "total_steps": 2048,
        }

        result = self.router.execute(order, deterministic=False)
        action_id = result["action_id"]
        action_name = {0: "market", 1: "join", 2: "improve", 3: "cancel"}[action_id]

        if self.notifier:
            msg = (
                f"📈 [{self.session_id}] {signal['symbol']} {signal['side'].upper()} "
                f"{signal['qty']} via {action_name.upper()} (mid={signal['mid']:.2f}, "
                f"spread={signal.get('spread',0.01):.4f})"
            )
            self.notifier.send_message(msg)

        # Place or simulate the order
        broker_resp = None
        if action_name == "cancel":
            broker_resp = {"status": "canceled"}
        else:
            broker_resp = self.broker.place_order(
                symbol=signal["symbol"],
                side=signal["side"],
                qty=signal["qty"],
                order_type="market" if action_name == "market" else "limit",
                limit_price=signal["mid"] if action_name != "market" else None,
            )

        # Merge result and broker response
        result["broker_resp"] = broker_resp
        result["timestamp"] = datetime.now(timezone.utc)
        result["action_name"] = action_name

        self._log_execution(result)

        # Telegram fill notification
        if self.notifier and result.get("filled"):
            self.notifier.send_message(
                f"✅ FILLED {signal['symbol']} {signal['side']} "
                f"{signal['qty']} @ {result['fill_price']:.2f} "
                f"lat={result['latency_ms']:.1f}ms"
            )

        return result

    def _log_execution(self, data: Dict[str, Any]):
        df = pd.DataFrame([data])
        df["session_id"] = self.session_id
        header = not os.path.exists(self.log_path)
        df.to_csv(self.log_path, mode="a", index=False, header=header)

    def summary(self) -> pd.DataFrame:
        if os.path.exists(self.log_path):
            return pd.read_csv(self.log_path)
        return pd.DataFrame()


# Example usage:
# router_cfg = RouterConfig()
# manager = LiveExecutionManager("models/phase17_exec_ppo.zip", router_cfg, dry_run=True)
# manager.execute_signal({"symbol":"AAPL","side":"buy","qty":10,"mid":190.2,"spread":0.02})
