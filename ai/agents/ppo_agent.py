# ai/agents/ppo_agent.py
# ===============================================================
#  PPOAgent wrapper for Stable Baselines3 PPO
#  Falls back to a safe stub if SB3/torch not available.
# ===============================================================

import os
import random

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.envs import DummyVecEnv
    SB3_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Torch/SB3 not available — running in STUB MODE ({e})")
    SB3_AVAILABLE = False


class PPOAgent:
    """
    PPOAgent handles reinforcement-learning based signal generation.
    If stable_baselines3 is not installed, runs in stub mode (random output).
    """

    def __init__(self, model_path: str = "models/ppo_agent.zip", load_existing: bool = True):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if SB3_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

        if SB3_AVAILABLE:
            if load_existing and os.path.exists(model_path):
                try:
                    self.model = PPO.load(model_path)
                    print(f"✅ PPO model loaded from {model_path}")
                except Exception as e:
                    print(f"⚠️ PPO load failed, creating new model: {e}")
                    self._init_new_model()
            else:
                self._init_new_model()
        else:
            # Stub fallback
            self.model = None

    def _init_new_model(self):
        if not SB3_AVAILABLE:
            return
        try:
            import gymnasium as gym
            env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
            self.model = PPO("MlpPolicy", env, verbose=0)
            print("✅ New PPO model initialized (dummy env)")
        except Exception as e:
            print(f"⚠️ PPO new model init failed: {e}")
            self.model = None

    def get_signal(self, symbol: str) -> float:
        """
        Returns trading signal in [-1, 1].
        In stub mode, returns random float; in SB3 mode, predicts based on dummy input.
        """
        if SB3_AVAILABLE and self.model is not None:
            try:
                import numpy as np
                obs = np.random.randn(1, 4)  # dummy observation
                action, _ = self.model.predict(obs, deterministic=True)
                return float(action[0] if isinstance(action, (list, tuple)) else action)
            except Exception:
                pass

        # Fallback (stub)
        return round(random.uniform(-1, 1), 3)
