from __future__ import annotations
import math, random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

@dataclass
class ExecEnvConfig:
    episode_minutes: int = 45
    tick_seconds: int = 1
    notional_per_episode: float = 100_000.0
    max_participation: float = 0.12
    max_spread_bps: int = 15
    slip_impact_scale: float = 1.0
    spread_reward_scale: float = 0.5
    nonfill_penalty: float = 0.2
    inventory_risk_bps: float = 3.0
    guardian_kill_switch: bool = True

class ExecutionEnv(gym.Env):
    """
    A lightweight microstructure execution simulator.
    Observations:
      [mid_ret, spread_bps, imb, vol_1s, vol_5s, depth_imb, rem_time, rem_notional_norm, pov_so_far]
    Actions (continuous box mapped inside step()):
      [slice_qty_factor (0..1), order_type_logits(3), urgency (0..1), limit_offset_norm (0..1)]
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: ExecEnvConfig, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        horizon = max(1, int((cfg.episode_minutes*60)//cfg.tick_seconds))
        self.horizon = horizon

        self.observation_space = spaces.Box(low=-10, high=10, shape=(9,), dtype=np.float32)
        # action: 1 scalar + 3 logits + 1 scalar + 1 scalar => 6 dims
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self._t = 0
        self._reset_episode_vars()

    def _reset_episode_vars(self):
        self._t = 0
        self.price = 100.0
        self.rem_notional = self.cfg.notional_per_episode
        self.base_volume_per_tick = 50_000  # notional/tick
        self.pov_so_far = 0.0
        self.last_mid = self.price
        self.position = 0.0
        self.pnl = 0.0
        self.killed = False

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_episode_vars()
        return self._obs(), {}

    # ---- synthetic microstructure helpers ----
    def _make_micro(self):
        # Random walk + mean reversion touch
        drift = self.rng.normal(0, 0.002)
        shock = self.rng.normal(0, 0.05)
        self.price = max(1e-3, self.price * (1.0 + drift + 0.1*math.tanh(-0.5*(self.price-100.0)/100.0) + shock/1000.0))
        spread_bps = np.clip(abs(self.rng.normal(7, 4)), 1, self.cfg.max_spread_bps)
        vol_1s = abs(self.rng.normal(0.02, 0.01))
        vol_5s = abs(self.rng.normal(0.05, 0.02))
        imb = np.clip(self.rng.normal(0, 0.5), -1, 1)
        depth_imb = np.clip(self.rng.normal(0, 0.6), -1, 1)
        return spread_bps, vol_1s, vol_5s, imb, depth_imb

    def _obs(self):
        spread_bps, vol_1s, vol_5s, imb, depth_imb = self._make_micro()
        mid_ret = (self.price - self.last_mid) / max(1e-6, self.last_mid)
        rem_time = 1.0 - (self._t / self.horizon)
        rem_notional_norm = self.rem_notional / max(1.0, self.cfg.notional_per_episode)
        obs = np.array([
            10*mid_ret, spread_bps/10.0, imb, vol_1s, vol_5s, depth_imb,
            rem_time, rem_notional_norm, self.pov_so_far
        ], dtype=np.float32)
        self.last_mid = self.price
        return obs

    def step(self, action):
        if self.killed:
            # forced no-op after kill switch
            self._t += 1
            terminated = (self._t >= self.horizon) or (self.rem_notional <= 1.0)
            return self._obs(), -0.01, terminated, False, {"killed": True}

        a = np.clip(action, 0.0, 1.0)
        slice_factor = a[0]                         # 0..1
        order_type_logits = a[1:4]                  # 3 logits -> type
        urgency = a[4]                              # 0..1 (interval)
        limit_offset_norm = a[5]                    # 0..1

        # decode order type
        t = np.argmax(order_type_logits)
        order_type = ["MARKET", "LIMIT", "POV"][t]

        # compute slice notional
        target_slice = slice_factor * 0.1 * self.cfg.notional_per_episode
        target_slice = min(target_slice, self.rem_notional)
        # POV cap
        tick_tape = self.base_volume_per_tick
        cap = self.cfg.max_participation * tick_tape
        notional_slice = float(np.clip(target_slice, 0, cap))
        pov = notional_slice / max(1.0, tick_tape)
        self.pov_so_far = 0.9*self.pov_so_far + 0.1*pov

        # price mechanics
        spread_bps, *_ = self._make_micro()
        half_spread = self.price * (spread_bps/10000)/2.0
        limit_off = (limit_offset_norm - 0.5) * 2 * (self.price * (5/10000))  # up to ±5 bps

        # fill model
        if order_type == "MARKET":
            exec_price = self.price + half_spread
            fill_ratio = 1.0
            impact = pov * 0.5 * (half_spread)  # crude
        elif order_type == "LIMIT":
            exec_price = self.price + min(half_spread, limit_off)
            # fill probability falls if price is aggressive against you
            fill_ratio = float(np.clip(0.7 + (limit_off/(half_spread+1e-6))*0.2, 0.1, 1.0))
            impact = pov * 0.25 * (half_spread)
        else:  # POV
            exec_price = self.price + 0.6*half_spread
            fill_ratio = float(np.clip(self.cfg.max_participation / max(1e-6, pov), 0.1, 1.0))
            impact = pov * 0.35 * (half_spread)

        filled_notional = notional_slice * fill_ratio
        slip = (exec_price - self.price)  # vs mid
        # rewards
        slippage = slip / self.price      # ~ bps in fractional
        spread_capture = (half_spread - (exec_price - self.price)) / self.price
        impact_pen = (impact / max(1e-6, self.price)) * self.cfg.slip_impact_scale

        reward = - slippage - impact_pen + self.cfg.spread_reward_scale*spread_capture

        # non-fill penalty to encourage finishing
        if filled_notional < notional_slice*0.5:
            reward -= self.cfg.nonfill_penalty * (1.0 - filled_notional/max(1e-6, notional_slice))

        # inventory/time risk: penalize remaining notional late in episode
        time_pressure = (self._t / self.horizon)
        inv_pen = time_pressure * (self.rem_notional / self.cfg.notional_per_episode) * (self.cfg.inventory_risk_bps/10000)
        reward -= inv_pen

        self.rem_notional -= filled_notional
        self.pnl -= filled_notional * slippage    # negative slippage reduces pnl

        # simple kill switch if too slow & late
        if self.cfg.guardian_kill_switch and time_pressure > 0.95 and self.rem_notional > 0.2*self.cfg.notional_per_episode:
            self.killed = True
            reward -= 0.5

        self._t += 1
        terminated = (self._t >= self.horizon) or (self.rem_notional <= 1.0) or self.killed
        return self._obs(), float(reward), terminated, False, {
            "exec_price": float(exec_price),
            "filled_notional": float(filled_notional),
            "order_type": order_type,
            "pov": float(pov),
            "rem_notional": float(self.rem_notional),
        }
