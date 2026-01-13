from .base import ExecutionPolicy
from typing import Dict, Any, Optional
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

class PPOExecutionPolicy(ExecutionPolicy):
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        if PPO is None:
            raise ImportError("stable_baselines3 not installed")
        self.model = PPO("MlpPolicy", **kwargs) if model_path is None else PPO.load(model_path)

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        obs = state["obs"] if "obs" in state else np.array([0,0,0,0,0,0], dtype=float)
        action, _ = self.model.predict(obs, deterministic=False)
        # Map continuous/discrete action to (size, aggression)
        size = max(int(abs(action[0]) * state.get("remaining_qty", 1)), 1)
        agg = ["passive","mid","market"][int(abs(action[1]) * 3) % 3] if action.shape[0] > 1 else "mid"
        return {"size": size, "aggression": agg, "price": None}
