import gymnasium as gym
import numpy as np

class LiveEnvGymAdapter(gym.Env):
    """
    Wraps your LiveTradingEnv to look like a Gym environment for PPO.
    Action space: discrete per symbol (0=hold, 1=buy, 2=sell) => vector of ints.
    Observation: flattened normalized feature vector from StateBuilder.
    """
    metadata = {"render.modes": []}

    def __init__(self, live_env, symbols):
        super().__init__()
        self.live_env = live_env
        self.symbols = symbols
        self.n_symbols = len(symbols)

        # Observation: use the state length from a sample
        sample_state = self.live_env.reset()
        obs_dim = sample_state.shape[0]
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)

        # Each symbol has 3 discrete actions; represent as a flat Discrete(3 * n_symbols) via MultiDiscrete
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_symbols)
        self._last_obs = sample_state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self.live_env.reset()
        self._last_obs = obs.astype(np.float32)
        return self._last_obs, {}

    def step(self, action_vec):
        # Convert vector [a0, a1, ...] to dict {sym: {side, confidence}}
        action_dict = {}
        for i, sym in enumerate(self.symbols):
            a = int(action_vec[i])
            side = "hold" if a == 0 else ("buy" if a == 1 else "sell")
            action_dict[sym] = {"side": side, "confidence": 0.55 if a != 0 else 0.5}
        next_obs, reward, done, info = self.live_env.step_once(action_dict)
        self._last_obs = next_obs.astype(np.float32)
        # Gym expects (obs, reward, terminated, truncated, info)
        return self._last_obs, float(reward), bool(done), False, info

    def render(self):
        return None
