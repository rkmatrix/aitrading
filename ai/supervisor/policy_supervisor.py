# ai/supervisor/policy_supervisor.py
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------- Contracts ----------

class PolicyAdapter(Protocol):
    """Minimal interface your policy wrappers should implement."""
    name: str

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return:
            {
              "targets": { "AAPL": 0.25, "MSFT": -0.10, ... },  # desired position in [-1, 1]
              "confidence": 0.0..1.0,
              "meta": {...}   # optional details like raw logits, timestamp, etc.
            }
        """
        ...

    def update(self, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        """Inform the policy of realized reward/feedback (optional; may be no-op)."""
        ...


@dataclass
class EWMA:
    alpha: float
    value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


@dataclass
class PolicyState:
    adapter: PolicyAdapter
    weight: float
    short_ewma: EWMA
    long_ewma: EWMA
    last_reward: float = 0.0
    last_ts: float = field(default_factory=time.time)

    def score(self, mode: str = "advantage") -> float:
        """
        Returns the scoring signal used for blending weight adaptation.
        - 'advantage': short - long  (captures recent out/under-performance)
        - 'short': short-only
        - 'long': long-only
        """
        s = self.short_ewma.value if self.short_ewma.value is not None else 0.0
        l = self.long_ewma.value if self.long_ewma.value is not None else 0.0
        if mode == "short":
            return s
        if mode == "long":
            return l
        return s - l  # default


@dataclass
class SupervisorConfig:
    symbols: List[str]
    policy_names: List[str]
    # EWMA alphas (converted from half-life or directly provided)
    short_alpha: float = 0.3
    long_alpha: float = 0.05
    adaptation_rate: float = 0.5      # 0..1, how aggressively to move current weights toward new normalized weights
    min_weight: float = 0.05          # floor for each policy
    max_weight: float = 0.85          # soft cap; if needed, we’ll re-normalize
    temperature: float = 1.0          # softmax temperature; lower = peakier
    scoring_mode: str = "advantage"   # 'advantage' | 'short' | 'long'
    confidence_power: float = 1.0     # raise each policy's confidence^p before blending
    normalize_confidence: bool = True
    rebalance_interval_s: float = 5.0
    reward_key: str = "reward"        # a key to look up per-policy reward in info (optional)


# ---------- Supervisor ----------

class MultiPolicySupervisor:
    """
    Tracks multiple live policies, maintains rolling performance,
    and produces adaptive blending weights.
    """
    def __init__(self, cfg: SupervisorConfig, adapters: List[PolicyAdapter]):
        # Filter to only configured policy names (order preserved)
        chosen = []
        for n in cfg.policy_names:
            found = next((a for a in adapters if a.name == n), None)
            if not found:
                logger.warning("⚠️ Supervisor: policy '%s' not provided by adapters list. Skipping.", n)
            else:
                chosen.append(found)

        if not chosen:
            raise ValueError("No matching policies provided to MultiPolicySupervisor.")

        self.cfg = cfg
        self._states: Dict[str, PolicyState] = {}

        # Uniform initial weights
        w0 = 1.0 / len(chosen)
        for a in chosen:
            st = PolicyState(
                adapter=a,
                weight=w0,
                short_ewma=EWMA(alpha=cfg.short_alpha),
                long_ewma=EWMA(alpha=cfg.long_alpha),
            )
            self._states[a.name] = st

        self._last_rebalance = 0.0
        logger.info("🧠 MultiPolicySupervisor initialized with %d policy(ies): %s",
                    len(self._states), ", ".join(self._states.keys()))

    # ---- Introspection ----
    @property
    def policies(self) -> List[str]:
        return list(self._states.keys())

    def get_weights(self) -> Dict[str, float]:
        return {k: st.weight for k, st in self._states.items()}

    def get_scores(self) -> Dict[str, float]:
        return {k: st.score(self.cfg.scoring_mode) for k, st in self._states.items()}

    # ---- Core loop helpers ----
    def predict_all(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for name, st in self._states.items():
            try:
                pred = st.adapter.predict(obs) or {}
                # Normalize/defend confidence
                conf = float(pred.get("confidence", 1.0))
                if self.cfg.normalize_confidence:
                    conf = min(max(conf, 0.0), 1.0)
                pred["confidence"] = conf
                # Ensure targets exist for every configured symbol (fill zeros)
                targets = {sym: float(pred.get("targets", {}).get(sym, 0.0)) for sym in self.cfg.symbols}
                pred["targets"] = targets
                out[name] = pred
            except Exception as e:
                logger.exception("💥 Policy '%s' predict failed: %s", name, e)
                out[name] = {"targets": {sym: 0.0 for sym in self.cfg.symbols}, "confidence": 0.0, "meta": {"error": str(e)}}
        return out

    def update_rewards(self, rewards: Dict[str, float], info_by_policy: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Update short/long EWMA and notify policies.
        If a policy is missing in rewards, we use 0.0 (neutral).
        """
        for name, st in self._states.items():
            r = float(rewards.get(name, 0.0))
            st.last_reward = r
            st.short_ewma.update(r)
            st.long_ewma.update(r)
            try:
                st.adapter.update(r, (info_by_policy or {}).get(name))
            except Exception as e:
                logger.warning("⚠️ Policy '%s' update() raised: %s", name, e)

    def maybe_rebalance(self) -> Dict[str, float]:
        now = time.time()
        if (now - self._last_rebalance) < self.cfg.rebalance_interval_s:
            return self.get_weights()
        self._last_rebalance = now

        scores = self.get_scores()
        names = list(scores.keys())
        x = np.array([scores[n] for n in names], dtype=float)

        # Temperatured softmax with numerical stability
        t = max(self.cfg.temperature, 1e-6)
        x_scaled = x / t
        x_shift = x_scaled - np.max(x_scaled)
        exp_x = np.exp(x_shift)
        base = exp_x / np.sum(exp_x) if np.isfinite(exp_x).all() else np.ones_like(x) / len(x)

        # Apply floors / caps, then renormalize
        base = np.clip(base, self.cfg.min_weight, self.cfg.max_weight)
        base = base / np.sum(base)

        # Smoothly adapt current weights toward base
        cur = np.array([self._states[n].weight for n in names], dtype=float)
        new = (1.0 - self.cfg.adaptation_rate) * cur + self.cfg.adaptation_rate * base
        new = np.clip(new, self.cfg.min_weight, self.cfg.max_weight)
        new = new / np.sum(new)

        for i, n in enumerate(names):
            self._states[n].weight = float(new[i])

        logger.info("🔀 Weights rebalanced → %s",
                    ", ".join(f"{n}={self._states[n].weight:.3f}" for n in names))
        logger.debug("Scores: %s", ", ".join(f"{n}={scores[n]:.4f}" for n in names))
        return self.get_weights()

    # Convenience: one-step update
    def step(self,
             obs: Dict[str, Any],
             realized_rewards: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        1) Get predictions from all policies
        2) If rewards provided, update rollups and rebalance weights
        3) Return {"predictions": {...}, "weights": {...}, "scores": {...}}
        """
        preds = self.predict_all(obs)
        if realized_rewards is not None:
            self.update_rewards(realized_rewards)
            self.maybe_rebalance()
        return {
            "predictions": preds,
            "weights": self.get_weights(),
            "scores": self.get_scores(),
        }
