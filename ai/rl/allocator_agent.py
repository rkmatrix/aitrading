"""
AllocatorRLAgent – Reinforcement Learning based Capital Allocator
(Used in Phase 30 Capital Allocator Coordinator)

Responsibilities:
    • Interface between PortfolioBrain & RL environment
    • Initialize / load RL model (Stable-Baselines3 PPO by default)
    • Predict optimal allocations (weights per asset)
    • Optionally train online or update periodically
"""

from __future__ import annotations
import os, logging
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
#   Allocator RL Agent
# ------------------------------------------------------------
class AllocatorRLAgent:
    def __init__(self, cfg: dict, env):
        """
        Initialize AllocatorRLAgent with configuration + environment.
        """
        self.cfg = cfg
        self.env = env
        self.model = None
        self.model_path = (
            cfg.get("agent", {}).get("model_path", "data/models/allocator_rl.zip")
        )

        algo = cfg.get("agent", {}).get("algo", "PPO").upper()
        policy_type = cfg.get("agent", {}).get("policy_type", "MlpPolicy")

        logger.info("🧠 AllocatorRLAgent initializing (algo=%s, policy=%s)", algo, policy_type)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # ✅ SAFER call (no KeyError if 'agent' missing)
        self.model = self._load_or_create(algo, policy_type)

    # --------------------------------------------------------
    def _load_or_create(self, algo: str, policy: str):
        """
        Load a saved RL model or create a new one if missing.
        """
        agent_cfg = self.cfg.get("agent", {}) or {}

        if os.path.exists(self.model_path):
            try:
                logger.info("📦 Loading pre-trained RL allocator model from %s", self.model_path)
                if algo == "PPO":
                    return PPO.load(self.model_path, env=self.env)
                elif algo == "A2C":
                    return A2C.load(self.model_path, env=self.env)
                elif algo == "DDPG":
                    return DDPG.load(self.model_path, env=self.env)
                elif algo == "SAC":
                    return SAC.load(self.model_path, env=self.env)
            except Exception as e:
                logger.warning("⚠️ Failed to load existing model: %s. Creating new one.", e)

        logger.info("🧩 Creating new %s allocator model.", algo)
        kwargs = dict(
            learning_rate=agent_cfg.get("learning_rate", 3e-4),
            gamma=agent_cfg.get("gamma", 0.99),
            verbose=0,
        )
        if algo == "PPO":
            return PPO(policy, self.env, **kwargs)
        elif algo == "A2C":
            return A2C(policy, self.env, **kwargs)
        elif algo == "DDPG":
            return DDPG(policy, self.env, **kwargs)
        elif algo == "SAC":
            return SAC(policy, self.env, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

    # --------------------------------------------------------
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """
        Predict the next allocation action given an observation.
        Returns (action, state_info)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        action, state = self.model.predict(observation, deterministic=deterministic)
        logger.debug("🧠 Predicted action=%s", str(action))
        return action, state

    # --------------------------------------------------------
    def train(self, timesteps: int = 10_000):
        """
        Train or fine-tune the allocator RL agent.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        logger.info("🎓 Training allocator RL model for %d timesteps…", timesteps)
        self.model.learn(total_timesteps=timesteps, progress_bar=True)
        self.model.save(self.model_path)
        logger.info("💾 Allocator model saved to %s", self.model_path)

    # --------------------------------------------------------
    def evaluate(self, n_episodes: int = 5):
        """
        Run evaluation episodes and compute average reward.
        """
        rewards = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_r = 0
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, truncated, _ = self.env.step(action)
                total_r += reward
                if done or truncated:
                    break
            rewards.append(total_r)
            logger.info("🎯 Eval episode %d reward=%.4f", ep + 1, total_r)
        avg_r = np.mean(rewards)
        logger.info("📊 Avg evaluation reward=%.4f", avg_r)
        return avg_r
