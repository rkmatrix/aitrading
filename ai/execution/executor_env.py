# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .reward_shaper import ExecutionReward, RewardConfig
from .feature_feeder import FeatureFeeder, FeatureConfig

class ExecutionStep(Tuple[np.ndarray, float, bool, dict]):
    pass

@dataclass
class EnvConfig:
    action_buckets: List[Tuple[float, float]] = None
    max_steps: int = 60
    tick_bps: float = 1.0
    fee_bps: float = 0.1
    slip_scale: float = 0.5

    def __post_init__(self):
        if self.action_buckets is None:
            self.action_buckets = [
                (0.00, 0.00),
                (0.05, 0.25),
                (0.10, 0.50),
                (0.20, 0.75),
                (0.35, 1.00),
            ]

class ExecutionEnv:
    """Toy execution environment. Replace price feed hookup for live/paper."""
    def __init__(self, env_cfg: EnvConfig, rew_cfg: RewardConfig, feat_cfg: FeatureConfig):
        self.env_cfg = env_cfg
        self.rew = ExecutionReward(rew_cfg)
        self.feat = FeatureFeeder(feat_cfg)
        self.reset()

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)

    def _sim_next_tick(self):
        drift = 0.0
        vol = 0.5 + np.abs(np.random.randn()) * 0.2
        shock = np.random.randn() * vol
        self.mid += shock * self.env_cfg.tick_bps
        self.volume = max(1.0, np.random.lognormal(mean=2.0, sigma=0.5))
        self.mids.append(self.mid)
        self.vols.append(self.volume)

    def reset(self, notional: float = 1_000_000.0, side: int = 1):
        self.t = 0
        self.side = side
        self.notional = float(notional)
        self.remaining = self.notional
        self.filled = 0.0
        self.mid = 0.0
        self.volume = 100.0
        self.mids: List[float] = [self.mid]
        self.vols: List[float] = [self.volume]
        return self._obs()

    def _obs(self):
        inventory_frac = self.remaining / max(self.notional, 1e-6)
        time_frac = self.t / max(self.env_cfg.max_steps - 1, 1)
        return self.feat.build_state(self.mids, self.vols, inventory_frac, time_frac)

    def step(self, action: int) -> ExecutionStep:
        slice_frac, agg = self.env_cfg.action_buckets[action]
        child_notional = self.remaining * slice_frac
        impact_bps = self.env_cfg.slip_scale * agg * (child_notional / (self.volume * 1000.0))
        fee_bps = self.env_cfg.fee_bps if child_notional > 0 else 0.0
        slippage_bps = impact_bps + fee_bps
        self.remaining -= child_notional
        self.filled += child_notional
        violated = self.t == (self.env_cfg.max_steps - 1) and self.remaining > 1e-6
        time_pen = 1.0
        self._sim_next_tick()
        self.t += 1
        done = self.t >= self.env_cfg.max_steps or self.remaining <= 1e-6
        reward = self.rew.compute(
            slippage_bps=slippage_bps,
            inventory_left=self.remaining / max(self.notional, 1e-6),
            time_penalty=time_pen,
            violated=violated,
        )
        info = {
            "child_notional": child_notional,
            "slippage_bps": slippage_bps,
            "fee_bps": fee_bps,
            "impact_bps": impact_bps,
            "filled": self.filled,
            "remaining": self.remaining,
        }
        return self._obs(), reward, done, info

    @property
    def state_dim(self) -> int:
        return self.feat.cfg.lookback * 2 + 2

    @property
    def action_dim(self) -> int:
        return len(self.env_cfg.action_buckets)
