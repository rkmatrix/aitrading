from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from .constraints import project_simplex


@dataclass
class AllocatorStep:
    state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict


class AllocatorEnv(gym.Env):
    """
    Environment producing allocation decisions each step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_assets: int,
        step_minutes: int = 5,
        action_mode: str = "softmax",
        temperature: float = 1.0,
        max_turnover_l1: float = 0.25,
        trust_region_l1: float = 0.15,
        simplex_project: bool = True,
        min_weight: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.step_minutes = step_minutes
        self.action_mode = action_mode
        self.temperature = temperature
        self.max_turnover_l1 = max_turnover_l1
        self.trust_region_l1 = trust_region_l1
        self.simplex_project = simplex_project
        self.min_weight = min_weight
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(-np.inf, np.inf, (256,), np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, (self.n_assets,), np.float32)

        self._t = 0
        self._last_w = np.full(self.n_assets, 1.0 / self.n_assets)
        self._episode_pnl = 0.0

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z / max(1e-6, self.temperature)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def reset(self, *, seed=None, options=None):
        if seed:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._last_w[:] = 1.0 / self.n_assets
        state = np.asarray(options.get("state", np.zeros(64)), dtype=np.float32)
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[: len(state)] = state
        return obs, {}

    def step(self, action: np.ndarray, info: Optional[Dict] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        info = info or {}
        logits = np.asarray(action, np.float64)
        w = self._softmax(logits) if self.action_mode == "softmax" else logits
        if self.simplex_project:
            w = project_simplex(w, self.min_weight)

        l1 = np.abs(w - self._last_w).sum()
        if l1 > self.max_turnover_l1:
            a = self.max_turnover_l1 / (l1 + 1e-12)
            w = a * w + (1 - a) * self._last_w
            w = project_simplex(w, self.min_weight)

        pnl = float(info.get("pnl", 0))
        costs = float(info.get("costs", 0))
        drawdown = float(info.get("drawdown", 0))
        kill = bool(info.get("kill", False))

        reward = pnl - costs
        self._episode_pnl += reward
        self._last_w = w
        self._t += 1

        next_state = np.asarray(info.get("state", np.zeros(64)), np.float32)
        obs = np.zeros(self.observation_space.shape, np.float32)
        obs[: len(next_state)] = next_state
        out = {"weights": w, "pnl": pnl, "costs": costs, "drawdown": drawdown}
        return obs, float(reward), kill, False, out

    def render(self):
        print(f"t={self._t} pnl_ep={self._episode_pnl:.4f} last_w={self._last_w}")
