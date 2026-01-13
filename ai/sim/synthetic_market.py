# ai/sim/synthetic_market.py
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple

import gymnasium as gym
import numpy as np

log = logging.getLogger(__name__)


class SyntheticCrashEnv(gym.Env):
    """
    Simple 1D price process with occasional crash events.
    Agent chooses position size; reward is PnL.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int = 256,
        crash_prob: float = 0.2,
        max_crash_drawdown: float = 0.4,
    ) -> None:
        super().__init__()
        self.episode_length = episode_length
        self.crash_prob = crash_prob
        self.max_crash_drawdown = max_crash_drawdown

        # Observations: [price_norm, t_frac]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32),
        )

        # Action: continuous position size in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._t = 0
        self._price = 1.0
        self._last_price = 1.0
        self._crashed = False

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._price = 1.0
        self._last_price = 1.0
        self._crashed = False
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        t_frac = self._t / max(self.episode_length, 1)
        return np.array([self._price, t_frac], dtype=np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        pos = float(action[0])

        # simulate return
        if (not self._crashed) and np.random.rand() < self.crash_prob:
            # crash event
            drawdown = np.random.uniform(0.1, self.max_crash_drawdown)
            ret = -drawdown
            self._crashed = True
        else:
            # small Gaussian noise
            ret = np.random.normal(loc=0.0005, scale=0.01)

        self._last_price = self._price
        self._price = max(0.1, self._price * (1.0 + ret))

        # pnl ~ position * return
        pnl = pos * ret

        obs = self._get_obs()
        terminated = self._t >= self.episode_length
        truncated = False
        info = {"pnl": pnl, "ret": ret, "pos": pos}
        return obs, float(pnl), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
