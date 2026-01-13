from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from ai.execution.execution_simulator import ExecutionSimulator, SimParams
from ai.execution.rewards import ExecReward

ACTION_MAP = {0: "market", 1: "join", 2: "improve", 3: "cancel"}

class ExecutionEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, df: pd.DataFrame, *, config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cfg = config
        self.max_steps = int(config.get("max_steps_per_episode", 2048))
        self.episode_length_minutes = int(config.get("episode_length_minutes", 120))
        self.randomize_start = bool(config.get("randomize_start", True))
        self.inventory_limit = float(config.get("inventory_limit", 500))
        self.base_cash = float(config.get("base_cash", 1_000_000))
        self.tick_size = float(config.get("tick_size", 0.01))
        self.allowed_actions: List[str] = config.get(
            "allowed_actions", ["market", "join", "improve", "cancel"]
        )

        sim_params = SimParams(
            tick_size=self.tick_size,
            passive_fill_prob_at_spread=float(config.get("passive_fill_prob_at_spread", 0.35)),
            improve_fill_boost=float(config.get("improve_fill_boost", 0.4)),
            join_fill_boost=float(config.get("join_fill_boost", 0.15)),
            a_bps=float(config.get("a_bps", 3.5)),
            sigma_bps=float(config.get("sigma_bps", 2.0)),
            market_slippage_extra_bps=float(config.get("market_slippage_extra_bps", 1.0)),
            spread_floor_bps=float(config.get("spread_floor_bps", 0.2)),
            fill_lat_ms_mean=float(config.get("fill_lat_ms_mean", 120.0)),
            fill_lat_ms_std=float(config.get("fill_lat_ms_std", 80.0)),
            cancel_prob=float(config.get("cancel_prob", 0.08)),
        )
        self.sim = ExecutionSimulator(sim_params)

        self.reward_fn = ExecReward(
            shortfall_weight=float(config.get("shortfall_weight", 1.0)),
            inventory_weight=float(config.get("inventory_weight", 5e-4)),
            latency_weight=float(config.get("latency_weight", 1e-3)),
            fill_bonus=float(config.get("fill_bonus", 0.05)),
        )

        # Observation: [spread_bps_scaled, side_sign, qty_norm, inventory_norm, time_left, mid_return_tanh]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.allowed_actions))

        self._t = 0
        self._idxs: np.ndarray | None = None
        self._episode_df: pd.DataFrame | None = None
        self._inventory = 0.0
        self._cash = self.base_cash
        self._last_mid = None

    def _sample_episode_indices(self) -> np.ndarray:
        df = self.df
        if self.randomize_start:
            sym = np.random.choice(df["symbol"].unique())
            sdf = df[df["symbol"] == sym]
        else:
            sdf = df
        ts = pd.to_datetime(sdf["timestamp"], utc=True)
        sdf = sdf.assign(_ts=ts).sort_values("_ts")
        if sdf.empty:
            raise RuntimeError("Empty dataframe for ExecutionEnv")
        start = np.random.randint(0, max(1, len(sdf) - 1))
        end = min(len(sdf), start + self.max_steps)
        return sdf.index.to_numpy()[start:end]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._idxs = self._sample_episode_indices()
        self._episode_df = self.df.loc[self._idxs].reset_index(drop=True)
        self._inventory = 0.0
        self._cash = self.base_cash
        self._last_mid = float(self._episode_df.loc[0, "mid"]) if "mid" in self._episode_df else 0.0
        obs = self._make_obs(row=self._episode_df.loc[0])
        return obs, {}

    def _make_obs(self, row: pd.Series) -> np.ndarray:
        mid = float(row.get("mid", 0.0) or 0.0)
        spread = float(row.get("spread", 0.0) or 0.0)
        qty = float(row.get("qty", 0.0) or 0.0)
        side = str(row.get("side", "buy"))
        side_sign = 1.0 if side == "buy" else -1.0
        spread_bps = (spread / max(1e-9, mid)) * 1e4 if mid > 0 else 0.0
        qty_norm = np.tanh(qty / max(1.0, self.inventory_limit))
        inv_norm = np.tanh(self._inventory / max(1.0, self.inventory_limit))
        time_left = 1.0 - (self._t / max(1.0, len(self._episode_df)))
        mid_ret = 0.0 if self._last_mid in (None, 0.0) or mid == 0.0 else np.tanh((mid - self._last_mid) / self._last_mid)
        return np.array([spread_bps/50.0, side_sign, qty_norm, inv_norm, time_left, mid_ret], dtype=np.float32)

    def step(self, action_id: int):
        assert self._episode_df is not None
        action = ACTION_MAP[int(action_id)]
        row = self._episode_df.loc[self._t]
        side_sign = 1 if row.get("side", "buy") == "buy" else -1
        qty = float(row.get("qty", 0.0) or 0.0)
        mid = float(row.get("mid", 0.0) or 0.0)
        spread = float(row.get("spread", 0.0) or 0.0)

        sim_out = self.sim.step(action=action, side_sign=side_sign, qty=qty, mid=mid, spread=spread)
        filled = sim_out["filled"]
        fill_price = sim_out["fill_price"]
        latency_ms = sim_out["latency_ms"]

        if filled and fill_price is not None:
            self._inventory += side_sign * qty
            self._cash -= side_sign * qty * fill_price

        reward = self.reward_fn(
            side=side_sign,
            mid_before=mid,
            fill_price=fill_price if filled else None,
            inventory=self._inventory,
            latency_ms=latency_ms,
            filled=bool(filled)
        )

        self._t += 1
        terminated = self._t >= len(self._episode_df)
        truncated = self._t >= self.max_steps

        if not terminated and not truncated:
            self._last_mid = float(self._episode_df.loc[self._t, "mid"]) if "mid" in self._episode_df else self._last_mid
            obs = self._make_obs(self._episode_df.loc[self._t])
        else:
            obs = self._make_obs(row)

        info = {
            "filled": filled,
            "fill_price": fill_price,
            "latency_ms": latency_ms,
            "inventory": self._inventory,
            "cash": self._cash,
            "action": action,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return f"t={self._t} inv={self._inventory:.1f} cash={self._cash:.2f}"
