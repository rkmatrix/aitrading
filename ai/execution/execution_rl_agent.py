# ai/execution/execution_rl_agent.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO

from ai.execution.online_finetune_bridge import OnlineFineTuneBridge

log = logging.getLogger("ExecutionRLAgent")


class ExecutionRLAgent:
    """
    FULL execution loop with Phase60 Online RL Fine-Tuning integrated.
    """

    def __init__(
        self,
        env,
        policy_path: str = "models/policies/EquityRLPolicy/current_policy/model.zip",
        finetune_cfg: str = "configs/phase60_online_finetune.yaml",
    ):
        self.env = env
        self.policy_path = Path(policy_path)

        # Load PPO model
        log.info("Loading current_policy → %s", self.policy_path)
        self.model: PPO = PPO.load(self.policy_path, print_system_info=False)

        # Phase 60 Online Fine-Tuning
        log.info("Initializing Online Fine-Tuning Bridge…")
        self.online_ft = OnlineFineTuneBridge(finetune_cfg)

        log.info("ExecutionRLAgent initialized with Phase60 online learning enabled.")

    # ==========================================================
    # Action selection (policy)
    # ==========================================================
    def act(self, obs: Any) -> float:
        action, _ = self.model.predict(obs, deterministic=True)
        return float(action)

    # ==========================================================
    # Single execution step (for sync loops)
    # ==========================================================
    def step_once(self, obs: np.ndarray):
        action = self.act(obs)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Phase 60 Online Fine-Tune: observe transitions
        self.online_ft.observe(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info,
        )

        return next_obs, reward, done, info

    # ==========================================================
    # Async real-time execution loop with PPO + Online RL
    # ==========================================================
    async def run(self, max_steps: int = 999999):
        log.info("🚀 Starting real-time PPO execution loop with online learning…")
        obs, info = self.env.reset()

        step = 0
        while step < max_steps:
            next_obs, reward, done, info = self.step_once(obs)

            # Update obs
            obs = next_obs
            step += 1

            if step % 10 == 0:
                log.info("Step %d | reward=%.4f", step, reward)

            if done:
                log.info("🔄 Episode done — resetting environment.")
                obs, info = self.env.reset()

            await asyncio.sleep(0.01)  # regulate execution speed

        log.info("Execution loop stopped after %d steps.", step)
