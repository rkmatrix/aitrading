# ai/online/alpha_online_trainer.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from .alpha_online_replay_env import ReplayTradingEnv

logger = logging.getLogger(__name__)


class AlphaOnlineTrainer:
    """
    AlphaOnlineTrainer

    - Loads PPO model from bundle path (SB3 'model.zip').
    - Wraps a ReplayTradingEnv around the provided transitions.
    - Runs a mini PPO.learn() update with reset_num_timesteps=False.
    """

    def __init__(
        self,
        *,
        policy_bundle_dir: str,
        model_file: str = "model.zip",
        cfg: Dict[str, Any],
    ) -> None:
        self.bundle_dir = Path(policy_bundle_dir)
        self.model_path = self.bundle_dir / model_file
        self.cfg = cfg

        if not self.model_path.exists():
            raise FileNotFoundError(f"PPO model file not found: {self.model_path}")

    def mini_update(
        self,
        transitions: List[Dict[str, Any]],
    ) -> Tuple[PPO, Dict[str, Any]]:
        """
        Run a small online training step on the given transitions.

        Returns:
            model (PPO): updated PPO instance
            info (dict): training info (e.g., avg_reward, n_transitions)
        """
        if not transitions:
            raise ValueError("AlphaOnlineTrainer.mini_update: empty transitions list")

        device = self.cfg.get("device", "cpu")
        train_timesteps = int(self.cfg.get("train_timesteps", 2048))
        verbose = int(self.cfg.get("verbose", 0))

        # Load current model from disk
        logger.info("🧠 AlphaOnlineTrainer: loading PPO from %s", self.model_path)
        model: PPO = PPO.load(self.model_path, device=device, print_system_info=False)

        # Prepare replay env with same spaces as the loaded model
        observation_space = model.observation_space
        action_space = model.action_space

        env = ReplayTradingEnv(
            transitions=transitions,
            observation_space=observation_space,
            action_space=action_space,
        )
        model.set_env(env)

        # Logging basic stats
        rewards = [float(t.get("reward", 0.0)) for t in transitions]
        avg_reward = float(np.mean(rewards)) if rewards else 0.0

        logger.info(
            "🎯 AlphaOnlineTrainer: starting mini PPO update | timesteps=%d | n_transitions=%d | avg_reward=%.4f",
            train_timesteps,
            len(transitions),
            avg_reward,
        )

        # PPO learn – continue training with same timestep counter
        model.learn(
            total_timesteps=train_timesteps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        logger.info("✅ AlphaOnlineTrainer: mini update completed")

        info = {
            "n_transitions": len(transitions),
            "avg_reward": avg_reward,
            "train_timesteps": train_timesteps,
        }
        return model, info
