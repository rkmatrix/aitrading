# ai/online/alpha_online_safety.py
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class AlphaOnlineSafety:
    """
    AlphaOnlineSafety

    - Estimates policy drift via KL(π_old || π_new) using sample observations.
    - If KL exceeds threshold, the update is rejected.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.kl_threshold: float = float(cfg.get("kl_threshold", 0.25))
        self.max_bad_kl_streak: int = int(cfg.get("max_bad_kl_streak", 3))
        self.bad_kl_streak: int = 0

    def _estimate_kl(
        self,
        old_model: PPO,
        new_model: PPO,
        obs_batch: np.ndarray,
    ) -> float:
        """
        Monte Carlo estimate: KL(π_old || π_new) ≈ E_{a~π_old}[ log π_old(a|s) - log π_new(a|s) ]
        """
        if obs_batch.ndim == 1:
            obs_batch = obs_batch[None, :]

        device = old_model.device
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32).to(device)

        with torch.no_grad():
            old_dist = old_model.policy.get_distribution(obs_tensor)
            new_dist = new_model.policy.get_distribution(obs_tensor)

            # sample actions from old distribution
            actions = old_dist.get_actions()
            old_logp = old_dist.log_prob(actions)
            new_logp = new_dist.log_prob(actions)

            kl = (old_logp - new_logp).mean().item()
        return float(kl)

    def check(
        self,
        *,
        old_model: PPO,
        new_model: PPO,
        obs_batch: np.ndarray,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Perform safety checks.

        Returns:
            ok (bool), info (dict)
        """
        if obs_batch.size == 0:
            logger.warning("AlphaOnlineSafety.check called with empty obs_batch; skipping KL check.")
            return True, {"reason": "no_obs"}

        kl_value = self._estimate_kl(old_model, new_model, obs_batch)

        info: Dict[str, Any] = {
            "kl_value": kl_value,
            "kl_threshold": self.kl_threshold,
        }

        if kl_value > self.kl_threshold:
            self.bad_kl_streak += 1
            logger.warning(
                "🚫 AlphaOnlineSafety: KL too high (%.4f > %.4f) [bad_streak=%d]",
                kl_value,
                self.kl_threshold,
                self.bad_kl_streak,
            )
            info["ok"] = False
            info["bad_kl_streak"] = self.bad_kl_streak
            return False, info

        # safe
        self.bad_kl_streak = 0
        logger.info(
            "🟢 AlphaOnlineSafety: KL check passed (%.4f <= %.4f)",
            kl_value,
            self.kl_threshold,
        )
        info["ok"] = True
        info["bad_kl_streak"] = self.bad_kl_streak
        return True, info
