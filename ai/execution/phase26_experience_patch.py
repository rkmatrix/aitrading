# ai/execution/phase26_experience_patch.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
import numpy as np

from ai.experience.replay_buffer import ReplayBuffer
from ai.execution.live_trade_sink import LiveTradeSink

# Optional learner for in-process continuous updates (kept lightweight)
try:
    from ai.agents.offpolicy_learner import OffPolicyLearner
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

class ExperiencePatch:
    """
    Phase 26 drop-in: initialize a ReplayBuffer + LiveTradeSink and (optionally) a tiny learner.
    Call `on_fill(...)` after each executed order when you have obs → action → next_obs.
    Optionally call `maybe_learn()` periodically for continuous updates.
    """
    def __init__(self, cfg_path: str, auto_learn: bool = True):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        bcfg = self.cfg["buffer"]
        self.buf = ReplayBuffer(
            capacity=bcfg["capacity"],
            prioritized=bcfg.get("prioritized", True),
            per_alpha=bcfg.get("per_alpha", 0.6),
            per_beta=bcfg.get("per_beta", 0.4),
            per_eps=bcfg.get("per_eps", 1e-6),
        )
        self.sink = LiveTradeSink(self.buf, self.cfg["reward"])
        self._auto_learn = auto_learn
        self._init_learner_done = False
        self._learn_every = int(self.cfg.get("learner", {}).get("grad_steps", 2000) // 10) or 100
        self._batch = int(self.cfg["buffer"].get("batch_size", 2048))
        self._tick = 0

    def _ensure_learner(self, obs_dim: int, act_dim: int):
        if not self._auto_learn or not _TORCH_OK:
            return
        if not hasattr(self, "learner"):
            lcfg = self.cfg["learner"]
            self.learner = OffPolicyLearner(
                obs_dim=obs_dim, act_dim=act_dim, lr=lcfg.get("lr", 3e-4), device=lcfg.get("device", "cpu")
            )
            self._init_learner_done = True

    def on_fill(
        self,
        obs: np.ndarray,
        action: np.ndarray | float | int,
        next_obs: np.ndarray,
        fill_dict: Dict[str, Any],
        exposure: float = 0.0,
    ) -> float:
        """
        Push one transition derived from live execution.
        Returns the computed reward for logging.
        """
        reward = self.sink.on_step(obs, action, next_obs, fill_dict, exposure)
        self._tick += 1

        # initialize learner lazily once we know dims
        if self._auto_learn and _TORCH_OK and not self._init_learner_done and len(self.buf) >= max(64, self._batch):
            idxs, w, (s, a, r, ns, d, info) = self.buf.sample(min(self._batch, len(self.buf)))
            self._ensure_learner(s.shape[1], a.shape[1])

        return reward

    def maybe_learn(self) -> dict | None:
        """
        If autolearn is enabled, consume a batch every `_learn_every` pushes.
        """
        if not (self._auto_learn and _TORCH_OK and hasattr(self, "learner")):
            return None
        if len(self.buf) < self._batch:
            return None
        if self._tick % self._learn_every != 0:
            return None

        idxs, w, batch = self.buf.sample(self._batch)
        metrics = self.learner.update(batch, w)
        # small PER refresh using critic loss as proxy for TD error
        self.buf.update_priorities(idxs, np.ones(len(idxs), dtype=np.float32) * max(1e-4, metrics["critic_loss"]))
        return metrics
