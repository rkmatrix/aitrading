# ai/adaptive/live_adaptor.py
from __future__ import annotations
from typing import Optional, Dict, Any, Callable
import numpy as np
from ai.adaptive.online_trainer import OnlineAdaptiveTrainer

class OnlineAdaptHook:
    """
    Thin adaptor that converts live events (state → action → fill) to RL transitions.
    Plug into your executor callbacks. If you only have fills, we approximate state/next_state
    via feature snapshots you pass when you issued the signal.
    """

    def __init__(self, trainer: OnlineAdaptiveTrainer):
        self.trainer = trainer
        self._pending: Dict[str, Dict[str, Any]] = {}  # keyed by client order id or symbol+ts

    def register_signal(self, key: str, state: np.ndarray, action_vec: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        """
        Call this when your agent emits a signal and before sending to broker.
        key: order id you generate (or symbol+timestamp string).
        state: feature vector at decision time.
        action_vec: encoded action used by env/agent (discrete index or continuous vector).
        meta: optional dict (e.g., {'symbol': 'AAPL', 'strategy':'PPO', 'volatility': 0.012})
        """
        self._pending[key] = {
            "s": np.asarray(state, dtype=np.float32),
            "a": np.asarray(action_vec, dtype=np.float32),
            "meta": meta or {},
        }

    def register_fill(self, key: str, next_state: np.ndarray, reward: float, done: bool, info: Optional[Dict[str, Any]] = None):
        """
        Call this when you receive a fill/close and have the realized reward for the step.
        If you don't have precise rewards yet, pass incremental PnL or risk-adjusted reward.
        """
        if key not in self._pending:
            # If unknown key, treat as one-off step (best effort)
            s = np.zeros_like(next_state, dtype=np.float32)
            a = np.zeros(1, dtype=np.float32)
            self.trainer.add_transition(s, a, float(reward), np.asarray(next_state, dtype=np.float32), bool(done), info or {})
            return

        rec = self._pending.pop(key)
        s = rec["s"]
        a = rec["a"]
        meta = rec.get("meta", {})
        merged_info = {}
        merged_info.update(meta)
        if info:
            merged_info.update(info)

        self.trainer.add_transition(
            s,
            a,
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            merged_info
        )
