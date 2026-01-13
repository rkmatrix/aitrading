# ai/execution/live_trade_sink.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from ai.experience.replay_buffer import Transition
from ai.experience.reward_memory import RewardMemory

class LiveTradeSink:
    """
    Glue that turns live fills + latest obs into ReplayBuffer transitions.
    Drop into your live loop after each step / fill event.
    """
    def __init__(self, buffer, reward_conf: dict):
        self.buf = buffer
        self.rm = RewardMemory(
            weights=reward_conf["weights"],
            pnl_norm_window=reward_conf.get("pnl_norm_window", 2000),
            dd_window=reward_conf.get("dd_window", 1000),
            clip=reward_conf.get("clip", None),
        )

    def on_step(self, obs, action, next_obs, fill: Dict[str, Any], exposure: float):
        """
        obs, next_obs: np.ndarray
        action: np.ndarray or scalar
        fill: dict with keys { realized_pnl, unrealized_pnl, tx_cost, ts, symbol, slippage, order_id, done }
        exposure: float
        """
        r = self.rm.step(
            realized_pnl=float(fill.get("realized_pnl", 0.0)),
            unrealized_pnl=float(fill.get("unrealized_pnl", 0.0)),
            tx_cost=float(fill.get("tx_cost", 0.0)),
            exposure=float(exposure),
        )
        tr = Transition(
            state=np.asarray(obs, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=r,
            next_state=np.asarray(next_obs, dtype=np.float32),
            done=bool(fill.get("done", False)),
            info={
                "ts": fill.get("ts"),
                "symbol": fill.get("symbol"),
                "slippage": fill.get("slippage", 0.0),
                "order_id": fill.get("order_id"),
                "realized_pnl": float(fill.get("realized_pnl", 0.0)),
                "unrealized_pnl": float(fill.get("unrealized_pnl", 0.0)),
                "tx_cost": float(fill.get("tx_cost", 0.0)),
            },
        )
        # Use abs(reward) as initial priority; learner can update later by TD error
        self.buf.push(tr, priority=abs(r))
        return r
