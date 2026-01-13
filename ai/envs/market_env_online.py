import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd


class MarketEnvOnline(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, cash: float = 100_000.0, fee_bps: float = 1.0, max_pos: int = 1):
        super().__init__()
        assert {"close"}.issubset(features.columns), "features must include price columns via your pipeline"
        self.df = features.copy()
        self.ptr = 0
        self.cash0 = cash
        self.cash = cash
        self.pos = 0  # -1 short, 0 flat, +1 long
        self.fee = fee_bps / 1e4
        self.max_pos = max_pos
        self.price_col = "close"

        feat_cols = [c for c in self.df.columns if c not in ("open", "high", "low", "close", "volume")]
        self.feat_cols = feat_cols
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feat_cols) + 3,), dtype=np.float32)
        # action: {-1, 0, +1}
        self.action_space = spaces.Discrete(3)
        self._last_val = self._portfolio_value(self._price(self.ptr))

    def _price(self, i):
        return float(self.df.iloc[i][self.price_col])

    def _portfolio_value(self, price):
        return self.cash + self.pos * price

    def _obs(self, i):
        f = self.df.iloc[i][self.feat_cols].astype(np.float32).values if self.feat_cols else np.zeros(0, dtype=np.float32)
        price = self._price(i)
        obs = np.concatenate([f, np.array([price, self.pos, self.cash], dtype=np.float32)])
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 0
        self.cash = self.cash0
        self.pos = 0
        self._last_val = self._portfolio_value(self._price(self.ptr))
        return self._obs(self.ptr), {}

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        action = max(0, min(2, int(action)))
        target_pos = [-1, 0, 1][action]

        price = self._price(self.ptr)
        # trade if position changes
        if target_pos != self.pos:
            trade_size = abs(target_pos - self.pos)
            fee = price * self.fee * trade_size
            self.cash += (self.pos * price)  # close old
            self.cash -= fee
            self.pos = target_pos
            self.cash -= (self.pos * price)  # open new

        done = False
        self.ptr += 1
        if self.ptr >= len(self.df) - 1:
            self.ptr = len(self.df) - 1
            done = True

        new_price = self._price(self.ptr)
        val = self._portfolio_value(new_price)
        reward = (val - self._last_val) / max(1.0, self._last_val) - 0.00005 * abs(self.pos)
        self._last_val = val
        info = {"value": val, "price": new_price, "pos": self.pos}
        return self._obs(self.ptr), float(reward), done, False, info
