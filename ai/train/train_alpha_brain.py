"""
ai/train/train_alpha_brain.py

Phase 88 — Train AlphaBrainPolicy (Alpha Brain 1.0)

Uses:
- TradingEnv-v0 as base environment
- AlphaBrainObsWrapper for extended features
- PPO (Stable-Baselines3) with MlpPolicy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ai.env.wrappers.alpha_brain_obs_wrapper import AlphaBrainObsWrapper

logger = logging.getLogger("Phase88TrainAlphaBrain")
logging.basicConfig(level=logging.INFO)


def build_env(env_id: str) -> gym.Env:
    """
    Create TradingEnv-v0 and wrap it with AlphaBrainObsWrapper, Monitor and DummyVecEnv.
    """
    try:
        base_env = gym.make(env_id)
    except Exception as e:
        raise RuntimeError(
            f"Phase88: Failed to create env '{env_id}'. Ensure it is registered."
        ) from e

    wrapped = AlphaBrainObsWrapper(base_env)
    monitored = Monitor(wrapped)
    vec_env = DummyVecEnv([lambda: monitored])
    return vec_env


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg: Dict[str, Any] = {}
    cfg["env_id"] = raw.get("env_id", "TradingEnv-v0")
    cfg["total_timesteps"] = int(raw.get("total_timesteps", 50_000))
    cfg["policy_name"] = raw.get("policy_name", "AlphaBrainPolicy")

    cfg["ppo"] = raw.get("ppo", {}) or {}
    cfg["paths"] = raw.get("paths", {}) or {}
    return cfg


def train_alpha_brain(config_path: str) -> None:
    cfg = load_config(config_path)

    env_id = cfg["env_id"]
    total_timesteps = cfg["total_timesteps"]
    policy_name = cfg["policy_name"]

    out_root = Path(cfg["paths"].get("output_dir", "models/policies/AlphaBrainPolicy"))
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Phase88: Using env '%s'", env_id)
    env = build_env(env_id)

    # PPO hyperparameters (can be tuned via config)
    ppo_cfg = cfg["ppo"]
    ppo_kwargs: Dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "gamma": float(ppo_cfg.get("gamma", 0.99)),
        "learning_rate": float(ppo_cfg.get("learning_rate", 3e-4)),
        "n_steps": int(ppo_cfg.get("n_steps", 2048)),
        "batch_size": int(ppo_cfg.get("batch_size", 64)),
        "n_epochs": int(ppo_cfg.get("n_epochs", 10)),
        "ent_coef": float(ppo_cfg.get("ent_coef", 0.0)),
        "clip_range": float(ppo_cfg.get("clip_range", 0.2)),
        "verbose": int(ppo_cfg.get("verbose", 1)),
    }

    logger.info("Phase88: PPO hyperparameters: %s", ppo_kwargs)

    model = PPO(**ppo_kwargs)

    logger.info(
        "Phase88: Training AlphaBrainPolicy '%s' for %d timesteps...",
        policy_name,
        total_timesteps,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = out_root / "model.zip"
    model.save(str(model_path))

    logger.info("Phase88: AlphaBrainPolicy saved to %s", model_path)
