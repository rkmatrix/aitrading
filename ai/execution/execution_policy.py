from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass

from stable_baselines3 import PPO


@dataclass
class ObsSpec:
    inventory_limit: float = 500.0


def make_obs_vector(*, mid: float, spread: float, qty: float, side: str,
                    inventory: float, t: int, total_steps: int,
                    last_mid: Optional[float], spec: ObsSpec) -> np.ndarray:
    """Mirror the observation construction used in ExecutionEnv._make_obs."""
    side_sign = 1.0 if side == "buy" else -1.0
    spread_bps = (spread / max(1e-9, mid)) * 1e4 if mid > 0 else 0.0
    qty_norm = np.tanh(qty / max(1.0, spec.inventory_limit))
    inv_norm = np.tanh(inventory / max(1.0, spec.inventory_limit))
    time_left = 1.0 - (t / max(1.0, float(total_steps)))
    if last_mid in (None, 0.0) or mid == 0.0:
        mid_ret = 0.0
    else:
        mid_ret = np.tanh((mid - last_mid) / last_mid)
    vec = np.array([spread_bps/50.0, side_sign, qty_norm, inv_norm, time_left, mid_ret], dtype=np.float32)
    return vec


class ExecutionPolicy:
    """Wrapper around a trained PPO policy for the ExecutionEnv."""
    def __init__(self, model: PPO, obs_spec: ObsSpec):
        self.model = model
        self.spec = obs_spec

    @classmethod
    def load(cls, path: str, *, obs_spec: ObsSpec) -> "ExecutionPolicy":
        model = PPO.load(path)
        return cls(model=model, obs_spec=obs_spec)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if obs.ndim == 1:
            obs_batch = obs[None, :]
        else:
            obs_batch = obs
        action, _ = self.model.predict(obs_batch, deterministic=deterministic)
        if isinstance(action, (list, tuple, np.ndarray)):
            return int(action[0])
        return int(action)
