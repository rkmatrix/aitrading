# ai/adaptive/online_trainer.py
from __future__ import annotations
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple, Optional, Dict, Any, List

import numpy as np

Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Dict[str, Any]]

class OnlineBuffer:
    """Rolling FIFO buffer for online experience."""
    def __init__(self, max_size: int = 20_000):
        self._buf: Deque[Transition] = deque(maxlen=max_size)

    def store(self, s, a, r, s2, done, info=None):
        self._buf.append((s, a, r, s2, done, info or {}))

    def sample(self, batch_size: int) -> List[Transition]:
        n = len(self._buf)
        if n == 0:
            return []
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        return [self._buf[i] for i in idx]

    def __len__(self):
        return len(self._buf)


class RegimeBufferManager:
    """
    Keeps separate buffers per regime (bull/bear/sideways) and a global buffer.
    If regime is unknown, falls back to global ("all").
    """
    def __init__(self, max_size_each: int = 12_000, max_size_all: int = 24_000):
        self.buffers: Dict[str, OnlineBuffer] = {
            "bull": OnlineBuffer(max_size_each),
            "bear": OnlineBuffer(max_size_each),
            "sideways": OnlineBuffer(max_size_each),
            "all": OnlineBuffer(max_size_all),
        }

    @staticmethod
    def _infer_regime(info: Optional[Dict[str, Any]]) -> str:
        if not info:
            return "all"
        if "regime" in info and info["regime"] in ("bull", "bear", "sideways"):
            return info["regime"]
        # Heuristic fallback from volatility and drift sign (if provided)
        vol = float(abs(info.get("volatility", 0.0)))
        drift = float(info.get("price_drift", 0.0))
        if vol < 0.005:
            return "sideways"
        if drift >= 0:
            return "bull"
        return "bear"

    def store(self, s, a, r, s2, done, info=None):
        regime = self._infer_regime(info)
        self.buffers["all"].store(s, a, r, s2, done, info)
        self.buffers[regime].store(s, a, r, s2, done, info)

    def sample(self, batch_size: int, regime: Optional[str] = None) -> List[Transition]:
        if regime and regime in self.buffers and len(self.buffers[regime]) > 0:
            return self.buffers[regime].sample(batch_size)
        # default to balanced mix across regimes
        chunks: List[Transition] = []
        per = max(1, batch_size // 3)
        for key in ("bull", "bear", "sideways"):
            chunks.extend(self.buffers[key].sample(per))
        remain = batch_size - len(chunks)
        if remain > 0:
            chunks.extend(self.buffers["all"].sample(remain))
        return chunks

    def size(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.buffers.items()}

    def __len__(self):
        return sum(len(v) for v in self.buffers.values())


class RewardNormalizer:
    """Online reward normalization with moving mean/std."""
    def __init__(self, momentum: float = 0.995, eps: float = 1e-8):
        self.m = 0.0
        self.s = 0.0
        self.momentum = momentum
        self.eps = eps
        self._n = 0

    def update(self, r: float) -> float:
        self._n += 1
        delta = r - self.m
        self.m += (1 - self.momentum) * delta
        self.s = self.momentum * self.s + (1 - self.momentum) * delta * (r - self.m)
        std = math.sqrt(max(self.s, self.eps))
        return (r - self.m) / (std + self.eps)


@dataclass
class DriftStats:
    """Simple drift detector using rolling raw rewards and realized volatility."""
    window: int = 900
    rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=900))
    vols: Deque[float] = field(default_factory=lambda: deque(maxlen=900))

    def update(self, reward: float, realized_vol: Optional[float] = None) -> Dict[str, float]:
        self.rewards.append(float(reward))
        if realized_vol is not None:
            self.vols.append(float(realized_vol))
        return {
            "avg_r": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "std_r": float(np.std(self.rewards)) if self.rewards else 0.0,
            "avg_vol": float(np.mean(self.vols)) if self.vols else 0.0,
        }


class OnlineAdaptiveTrainer:
    """
    Regime-aware online trainer. The agent must implement:
      - adaptive_update(batch, epochs, lr_scale, kl_target, ewc_strength=None, ewc_state=None)
      - ewc_estimate_fisher(dataset)  (optional, for periodic consolidation)
    """
    def __init__(
        self,
        agent,
        buffer_max_each: int = 12_000,
        buffer_max_all: int = 24_000,
        batch_size: int = 2048,
        update_every_steps: int = 300,
        min_all_for_update: int = 5_000,
        max_online_epochs: int = 4,
        kl_target: float = 0.02,
        base_lr: float = 3e-5,
        lr_vol_alpha: float = 0.6,
        ewc_every_updates: int = 8,
        ewc_strength: float = 0.0,  # 0 disables EWC penalty; try 1e-3..1e-2 to enable
        telegram_hook=None,
        logger=None,
    ):
        self.agent = agent
        self.buffers = RegimeBufferManager(max_size_each=buffer_max_each, max_size_all=buffer_max_all)
        self.rnorm = RewardNormalizer(momentum=0.995)
        self.drift = DriftStats()
        self.batch_size = batch_size
        self.update_every_steps = update_every_steps
        self.min_all_for_update = min_all_for_update
        self.max_online_epochs = max_online_epochs
        self.kl_target = kl_target
        self.base_lr = base_lr
        self.lr_vol_alpha = lr_vol_alpha
        self.ewc_every_updates = ewc_every_updates
        self.ewc_strength = ewc_strength
        self.telegram = telegram_hook
        self.logger = logger
        self._step = 0
        self._updates = 0
        self._lock = threading.Lock()
        self._ewc_state = None  # (params_snapshot, fisher_diag)

    def _maybe_log(self, **kwargs):
        if self.logger is not None:
            try:
                self.logger.log_adaptive(**kwargs)
            except Exception:
                pass

    def _compute_lr_scale(self, avg_vol: float) -> float:
        if avg_vol <= 0:
            return 1.0
        scale = math.exp(-self.lr_vol_alpha * avg_vol)
        return float(np.clip(scale, 0.2, 1.0))

    def add_transition(self, s, a, r, s2, done, info=None):
        r_n = self.rnorm.update(r)
        self.buffers.store(s, a, r_n, s2, done, info)
        stats = self.drift.update(reward=r, realized_vol=(info or {}).get("volatility"))
        self._step += 1

        sizes = self.buffers.size()
        self._maybe_log(kind="collect", step=self._step, reward=r, reward_norm=r_n,
                        avg_r=stats["avg_r"], avg_vol=stats["avg_vol"],
                        buf_all=sizes.get("all", 0), buf_bull=sizes.get("bull", 0),
                        buf_bear=sizes.get("bear", 0), buf_side=sizes.get("sideways", 0))

        if (self._step % self.update_every_steps == 0) and (sizes.get("all", 0) >= self.min_all_for_update):
            return self._adaptive_update_cycle(stats)
        return None

    def _adaptive_update_cycle(self, stats: Dict[str, float]) -> Dict[str, Any]:
        with self._lock:
            regime = None
            avg_r, avg_vol = stats.get("avg_r", 0.0), stats.get("avg_vol", 0.0)
            if avg_vol > 0.02:
                regime = "bear"
            elif avg_r < 0:
                regime = "sideways"

            batch = self.buffers.sample(self.batch_size, regime=regime)
            if not batch:
                return {}

            lr_scale = self._compute_lr_scale(avg_vol)
            results = self.agent.adaptive_update(
                batch=batch,
                epochs=self.max_online_epochs,
                lr_scale=lr_scale,
                kl_target=self.kl_target,
                ewc_strength=(self.ewc_strength if self._ewc_state is not None else 0.0),
                ewc_state=self._ewc_state,
            )

            self._updates += 1

            if self.ewc_strength > 0.0 and (self._updates % self.ewc_every_updates == 0):
                ref_batch = self.buffers.sample(self.batch_size, regime=None)
                if hasattr(self.agent, "ewc_estimate_fisher"):
                    try:
                        self._ewc_state = self.agent.ewc_estimate_fisher(ref_batch)
                    except Exception:
                        pass

            if self.telegram:
                try:
                    msg = (f"🧠 Online update #{self._updates} @ step {self._step}\n"
                           f"• regime_focus={regime or 'mix'}  batch={len(batch)}  epochs={self.max_online_epochs}\n"
                           f"• lr_scale={lr_scale:.3f}  kl={results.get('kl', 0):.4f}\n"
                           f"• policy_loss={results.get('policy_loss', 0):.6f}  "
                           f"value_loss={results.get('value_loss', 0):.6f}")
                    self.telegram(msg)
                except Exception:
                    pass

            self._maybe_log(kind="update", step=self._step, updates=self._updates,
                            regime=regime or "mix", **results,
                            lr_scale=lr_scale, avg_r=avg_r, avg_vol=avg_vol)
            return results
