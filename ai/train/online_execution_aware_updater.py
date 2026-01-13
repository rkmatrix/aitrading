"""
ai/train/online_execution_aware_updater.py

Phase 81 — Online Execution-Aware Policy Update

Fine-tunes the existing EquityRLPolicyExecAware PPO model on TradingEnv-v0
using the same execution-aware reward shaping as Phase 76.

This is an OFFLINE script:
    • Does NOT place orders
    • Does NOT touch SmartOrderRouter
    • Only updates the policy weights on disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO

import ai.env  # registers TradingEnv-v0
from ai.env.wrappers.execution_aware_reward_wrapper import ExecutionAwareRewardWrapper
from ai.reward.execution_aware_reward import ExecutionAwareReward


logger = logging.getLogger(__name__)


def build_env(env_id: str, cfg: Dict[str, Any]) -> gym.Env:
    """
    Build TradingEnv-v0 and wrap with ExecutionAwareRewardWrapper.
    """
    reward_cfg = cfg.get("reward", {}) or {}

    rew = ExecutionAwareReward(
        slippage_weight=reward_cfg.get("slippage_weight", -1.0),
        spread_weight=reward_cfg.get("spread_weight", -0.5),
        latency_weight=reward_cfg.get("latency_weight", -0.0005),
        fill_prob_weight=reward_cfg.get("fill_prob_weight", 1.0),
        clamp_penalty=reward_cfg.get("clamp_penalty", -0.3),
        pred_block_penalty=reward_cfg.get("pred_block_penalty", -2.0),
        risk_block_penalty=reward_cfg.get("risk_block_penalty", -3.0),
    )

    env = gym.make(env_id)
    env = ExecutionAwareRewardWrapper(env, rew)
    return env


def online_update_execution_aware_policy(config_path: str) -> None:
    """
    Entry point for Phase 81.

    Steps:
      1) Load .env
      2) Load YAML config
      3) Build env (TradingEnv-v0 + execution-aware reward)
      4) Load existing PPO model from base_model_path
      5) Run extra .learn(total_timesteps)
      6) Save updated model to out_model_path
    """
    load_dotenv()
    logger.info("📄 Loading Phase 81 config from %s", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    env_id = cfg.get("env_id", "TradingEnv-v0")
    base_model_path = cfg.get("base_model_path", "models/policies/EquityRLPolicyExecAware/model.zip")
    out_model_path = cfg.get("out_model_path", base_model_path)
    total_timesteps = int(cfg.get("total_timesteps", 20000))

    ppo_cfg = cfg.get("ppo", {}) or {}

    logger.info("Phase81: env_id=%s", env_id)
    logger.info("Phase81: base_model_path=%s", base_model_path)
    logger.info("Phase81: out_model_path=%s", out_model_path)
    logger.info("Phase81: total_timesteps=%d", total_timesteps)

    env = build_env(env_id, cfg)

    # Ensure model file exists
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"Base model not found at {base_model_path}")

    logger.info("Phase81: Loading PPO model from %s ...", base_model_path)
    model: PPO = PPO.load(base_model_path, env=env)
    logger.info("Phase81: Model loaded. Starting online fine-tuning...")

    # Adjust some hyperparameters if provided
    if "learning_rate" in ppo_cfg:
        lr = float(ppo_cfg["learning_rate"])
        logger.info("Phase81: Overriding learning_rate -> %f", lr)
        # stable-baselines3 allows schedule or float; we use float
        model.lr_schedule = lambda _: lr

    # gamma can't be easily changed post-init; if user changes it, we log a warning
    if "gamma" in ppo_cfg:
        logger.info(
            "Phase81: gamma in config is %s, but cannot be changed on a loaded model; "
            "it will take effect only on a full retrain.",
            ppo_cfg["gamma"],
        )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    out_path = Path(out_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model_path)
    logger.info("✅ Phase81: Updated model saved to %s", out_model_path)
