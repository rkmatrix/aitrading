# ai/online/alpha_online_replay_env.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym

logger = logging.getLogger(__name__)


class ReplayTradingEnv(gym.Env):
    """
    ReplayTradingEnv

    A lightweight Env wrapper over an in-memory list of transitions.
    It *ignores* the agent's action and simply replays rewards/next_obs from data.

    Transitions format:
        row["obs"]      -> observation at time t
        row["action"]   -> action taken at time t (for logging only)
        row["reward"]   -> reward at t
        row["done"]     -> bool
        row["next_obs"] -> observation at t+1 (if missing, fallback to obs)

    The env terminates at end of list (truncated=True).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions: Sequence[Dict[str, Any]],
        observation_space,
        action_space,
    ) -> None:
        super().__init__()
        self.transitions: List[Dict[str, Any]] = list(transitions)
        if not self.transitions:
            raise ValueError("ReplayTradingEnv requires at least one transition")

        self.observation_space = observation_space
        self.action_space = action_space

        self._index: int = 0
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._index = 0

        first = self.transitions[0]
        obs = np.array(first["obs"], dtype=np.float32)
        self._last_obs = obs
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action):
        if self._index >= len(self.transitions):
            # Should not happen if PPO respects truncation, but be safe.
            logger.warning("ReplayTradingEnv.step called past end of transitions")
            obs = np.array(self._last_obs if self._last_obs is not None else self.transitions[-1]["obs"], dtype=np.float32)
            return obs, 0.0, True, True, {"warning": "past_end"}

        row = self.transitions[self._index]

        reward = float(row.get("reward", 0.0))
        done = bool(row.get("done", False))
        next_obs = row.get("next_obs", row.get("obs"))
        next_obs = np.array(next_obs, dtype=np.float32)

        self._last_obs = next_obs
        self._index += 1

        terminated = done
        truncated = self._index >= len(self.transitions)

        info = {
            "dataset_action": row.get("action"),
            "replay_index": self._index - 1,
        }
        return next_obs, reward, terminated, truncated, info

    def render(self):
        # no-op for this env
        return None

    def close(self):
        return None
