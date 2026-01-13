"""
ai/train/train_execution_aware_policy.py

Phase 76 — Train an execution-aware RL policy using:
    - ExecutionAwareReward (Phase 75)
    - ExecutionAwareRewardWrapper
    - PPO (stable-baselines3)

Policy will be saved under:
    models/policies/<policy_name>/model.zip

This version is updated to ALWAYS use the configured env_id
(e.g. 'TradingEnv-v0') and will NOT silently fall back to DummyTradingEnv.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import yaml

from stable_baselines3 import PPO

from ai.reward.execution_aware_reward import ExecutionAwareReward
from ai.env.wrappers.execution_aware_reward_wrapper import ExecutionAwareRewardWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Optional Dummy env (kept only for manual experimentation, not used
# in normal Phase 76 training anymore).
# ---------------------------------------------------------------------

class DummyTradingEnv(gym.Env):
    """
    Minimal dummy env for experimentation only.
    NOT used in default Phase 76 training when env_id is set.
    """
    metadata = {"render_modes": []}

    def __init__(self, n_assets: int = 3, episode_len: int = 128):
        super().__init__()
        self.n_assets = n_assets
        self.episode_len = episode_len
        self.step_count = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_assets * 4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # Random walk-style observation & reward
        obs = np.random.randn(*self.observation_space.shape).astype(np.float32)
        pnl = float(np.random.randn() * 0.01)  # fake PnL-ish reward

        terminated = self.step_count >= self.episode_len
        truncated = False

        # Fake execution info
        info = {
            "execution": {
                "slippage": float(np.random.randn() * 0.02),
                "spread": float(abs(np.random.randn()) * 0.01),
                "latency_ms": float(abs(np.random.randn()) * 1000),
            },
            "prediction": {
                "pred_fill_prob": float(np.clip(np.random.rand(), 0.0, 1.0)),
            },
            "routing": {
                "clamped_by_prediction": bool(np.random.rand() < 0.1),
                "prediction_blocked": False,
                "risk_blocked": False,
            },
        }

        return obs, pnl, terminated, truncated, info


# ---------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------

def build_env(env_id: str | None) -> gym.Env:
    """
    Build the environment for Phase 76.

    For your bot, we REQUIRE a real env_id (e.g. 'TradingEnv-v0').
    We do NOT silently fall back to DummyTradingEnv anymore.

    If you want to use DummyTradingEnv manually, you can pass
    env_id=None and call DummyTradingEnv() yourself.
    """
    if not env_id:
        raise ValueError(
            "Phase76: env_id is not set. Please configure env.id "
            "in configs/phase76_train_execution_aware.yaml to 'TradingEnv-v0'."
        )

    try:
        env = gym.make(env_id)
        logger.info("Phase76: Using env '%s' (observation_space=%s, action_space=%s)",
                    env_id, env.observation_space, env.action_space)
        return env
    except Exception as e:
        raise RuntimeError(
            f"Phase76: Failed to create env '{env_id}'. "
            f"Ensure TradingEnv-v0 is properly registered. Original error: {e}"
        ) from e


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_execution_aware_policy(config_path: str) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Phase76 config not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    env_id = cfg.get("env", {}).get("id", None)
    env = build_env(env_id)

    # Build execution-aware reward engine from config
    exec_rew_cfg = cfg.get("env", {}).get("execution_reward", {}) or {}
    reward_engine = ExecutionAwareReward(
        slippage_weight=float(exec_rew_cfg.get("slippage_weight", -1.0)),
        spread_weight=float(exec_rew_cfg.get("spread_weight", -0.5)),
        latency_weight=float(exec_rew_cfg.get("latency_weight", -0.0005)),
        fill_prob_weight=float(exec_rew_cfg.get("fill_prob_weight", 1.0)),
        clamp_penalty=float(exec_rew_cfg.get("clamp_penalty", -0.3)),
        pred_block_penalty=float(exec_rew_cfg.get("pred_block_penalty", -2.0)),
        risk_block_penalty=float(exec_rew_cfg.get("risk_block_penalty", -3.0)),
    )

    wrapped_env = ExecutionAwareRewardWrapper(env, reward_engine)

    train_cfg = cfg.get("train", {}) or {}
    total_timesteps = int(train_cfg.get("total_timesteps", 100_000))
    policy = train_cfg.get("policy", "MlpPolicy")

    # PPO hyperparams
    ppo_kwargs = dict(
        policy=policy,
        env=wrapped_env,
        gamma=float(train_cfg.get("gamma", 0.99)),
        learning_rate=float(train_cfg.get("learning_rate", 3.0e-4)),
        n_steps=int(train_cfg.get("n_steps", 2048)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        ent_coef=float(train_cfg.get("ent_coef", 0.0)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        verbose=1,
    )

    logger.info("Phase76: PPO hyperparameters: %s", ppo_kwargs)

    model = PPO(**ppo_kwargs)

    # Logging directory if you use tensorboard later
    out_cfg = cfg.get("output", {}) or {}
    log_dir = Path(out_cfg.get("log_dir", "logs/phase76"))
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SB3_LOGDIR", str(log_dir))

    policy_name = out_cfg.get("policy_name", "EquityRLPolicyExecAware")
    base_dir = Path(out_cfg.get("base_dir", "models/policies"))
    policy_dir = base_dir / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "model.zip"

    logger.info("Phase76: Training execution-aware policy '%s' ...", policy_name)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    logger.info("Phase76: Saving policy to %s", policy_path)
    model.save(str(policy_path))

    logger.info("Phase76: Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_execution_aware_policy("configs/phase76_train_execution_aware.yaml")
