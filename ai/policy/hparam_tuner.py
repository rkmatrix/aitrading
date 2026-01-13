# ai/policy/hparam_tuner.py
"""
Phase 101.5 — PPO Hyperparameter Tuner for Multi-Symbol OHLCV Environment.

Uses Optuna to search PPO hyperparameters:
    • learning_rate
    • n_steps
    • batch_size
    • gamma
    • gae_lambda
    • clip_range
    • ent_coef

Objective:
    Maximize final equity after short PPO training runs (2k–10k steps)
    Penalize volatility & drawdown.

Outputs:
    data/hparams/best_p101.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import optuna
import gymnasium as gym
from stable_baselines3 import PPO

from ai.env.env_registration import register_env

log = logging.getLogger("HparamTuner")


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------
# Objective function
# ---------------------------------------------------------
def make_objective(env_id: str, symbols: List[str], trial_steps: int = 3000):

    def objective(trial: optuna.Trial) -> float:
        register_env()

        env = gym.make(env_id, symbols=symbols)

        # Sample hyperparameters
        hparams = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 3e-3),
            "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
        }

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=None,
            **hparams
        )

        model.learn(total_timesteps=trial_steps)

        # Evaluate with one rollout episode
        obs, _ = env.reset()
        done = False
        truncated = False
        equity_start = 1.0
        equity = 1.0
        peak = 1.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            equity = info.get("equity", equity)
            peak = info.get("peak_equity", peak)

        # Metrics
        dd = max((peak - equity) / peak, 0.0)
        score = equity - (0.2 * dd)  # reward good equity, penalize DD

        trial.set_user_attr("final_equity", float(equity))
        trial.set_user_attr("drawdown", float(dd))

        return float(score)

    return objective


# ---------------------------------------------------------
# Tuner class
# ---------------------------------------------------------
class PPOHyperparamTuner:
    """
    Wrapper around Optuna for Phase 101 hyperparameter optimization.
    """

    def __init__(self, env_id: str, symbols: List[str]) -> None:
        self.env_id = env_id
        self.symbols = symbols
        self.hparam_dir = ensure_dir("data/hparams")
        self.hparam_path = self.hparam_dir / "best_p101.json"

    def run(self, n_trials: int = 20, trial_steps: int = 3000):
        log.info("🎯 Phase 101.5 Hyperparameter Search starting...")

        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(self.env_id, self.symbols, trial_steps),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_params
        best["score"] = float(study.best_value)
        best["final_equity"] = float(study.best_trial.user_attrs.get("final_equity", 1.0))
        best["drawdown"] = float(study.best_trial.user_attrs.get("drawdown", 0.0))

        self.hparam_path.write_text(json.dumps(best, indent=2))

        log.info(f"📈 Best hyperparameters saved → {self.hparam_path}")
        return best
