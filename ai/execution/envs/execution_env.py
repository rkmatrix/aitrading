import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ExecutionEnv(gym.Env):
    """State: [spread, imbalance, vol, remaining_time, remaining_qty, micro_alpha]
       Action: [size_bucket, aggression_bucket] (discrete) or continuous mapped outside."""
    metadata = {"render_modes": []}

    def __init__(self, simulator, cost_model, horizon_steps:int=100, max_qty:int=10_000, seed:int=42):
        super().__init__()
        self.sim = simulator
        self.cost = cost_model
        self.horizon = horizon_steps
        self.max_qty = max_qty
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.MultiDiscrete([10, 3])  # size bucket, aggression
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self._reset_state()

    def _reset_state(self):
        self.t = 0
        self.remaining_qty = self.max_qty
        self.obs = np.zeros(6, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._reset_state()
        o = self._observe()
        return o, {}

    def _observe(self):
        lob = self.sim.snapshot()
        spread = lob["spread"]
        imbalance = lob["imbalance"]
        vol = lob["volatility"]
        micro_alpha = lob.get("micro_alpha", 0.0)
        remaining_time = 1.0 - self.t / max(self.horizon,1)
        remaining_qty = self.remaining_qty / max(self.max_qty,1)
        self.obs[:] = [spread, imbalance, vol, remaining_time, remaining_qty, micro_alpha]
        return self.obs

    def step(self, action):
        size_bucket, agg_bucket = int(action[0]), int(action[1])
        child_qty = max(int((size_bucket+1)/10 * self.remaining_qty), 1)
        fill_px, fill_qty, lob = self.sim.execute(child_qty, agg_bucket)
        # cost in currency terms:
        c = self.cost.estimate(
            side=+1, qty=fill_qty, price=lob["mid"], spread=lob["spread"],
            participation=lob.get("participation", 0.1), volatility=lob["volatility"], latency_ms=lob.get("latency_ms",25)
        )
        impl_shortfall = (fill_px - lob["arrival_price"]) * fill_qty  # buy side
        penalty_unfilled = 0.0
        self.remaining_qty -= fill_qty
        self.t += 1
        terminated = (self.t >= self.horizon) or (self.remaining_qty <= 0)
        if terminated and self.remaining_qty > 0:
            penalty_unfilled = 0.001 * self.remaining_qty  # small penalty; tune
        reward = - (impl_shortfall + c.total + penalty_unfilled)
        obs = self._observe()
        return obs, float(reward), terminated, False, {"fill_qty": fill_qty, "fill_px": fill_px}
