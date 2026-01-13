"""
Phase26 Adaptive Execution Brain – RL PPO policy bridge for live trading.
"""
from __future__ import annotations
import numpy as np
from ai.execution_rl.agent.exe_ppo_agent import ExecutionRLAgent
from ai.execution_rl.envs.execution_env import ExecutionEnv, ExecEnvConfig


class AdaptiveExecutionBrain:
    """
    Loads a trained Phase26 PPO policy and produces trade intents
    based on live tick features.
    """

    def __init__(self, model_path: str, seed: int = 42):
        self.env = ExecutionEnv(ExecEnvConfig(), seed=seed)
        self.agent = ExecutionRLAgent(lambda: self.env, {}, out_dir="artifacts/phase26/tmp")
        self.agent.load(model_path)
        self.last_action = None

    def decide(self, features: dict) -> dict:
        """
        Convert live tick-level features into 9D obs vector and return action dict.
        """
        obs = np.array([
            features.get("mid_ret", 0.0),
            features.get("spread_bps", 0.0) / 10.0,
            features.get("imbalance", 0.0),
            features.get("vol_1s", 0.0),
            features.get("vol_5s", 0.0),
            features.get("depth_imb", 0.0),
            features.get("rem_time", 1.0),
            features.get("rem_notional_norm", 1.0),
            features.get("pov_so_far", 0.0),
        ], dtype=np.float32)

        action = self.agent.act(obs)
        self.last_action = action

        # Decode the action the same way as the simulation env
        slice_factor = float(action[0])
        order_type = ["MARKET", "LIMIT", "POV"][int(np.argmax(action[1:4]))]
        urgency = float(action[4])
        limit_off_norm = float(action[5])

        return {
            "slice_factor": slice_factor,
            "order_type": order_type,
            "urgency": urgency,
            "limit_off_norm": limit_off_norm,
        }
