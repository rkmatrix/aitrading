from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Protocol, runtime_checkable

import yaml  # type: ignore

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyLike(Protocol):
    def predict(self, obs: Any) -> Any:
        ...


@dataclass
class PolicyConfig:
    name: str
    base_weight: float


@dataclass
class BlendConfig:
    rl_confidence_boost: float = 0.5
    high_vol_boost_momentum: float = 0.2
    low_vol_boost_trend: float = 0.2
    min_total_weight: float = 0.001


class PolicyBlender:
    """
    Real-time blender for combining multiple policy outputs.
    """

    def __init__(
        self,
        policies: Dict[str, PolicyLike],
        policy_cfgs: List[PolicyConfig],
        blend_cfg: BlendConfig,
    ) -> None:
        self.policies = policies
        self.policy_cfgs = policy_cfgs
        self.blend_cfg = blend_cfg

    @classmethod
    def from_yaml(cls, cfg_path: str, policies: Dict[str, PolicyLike]) -> "PolicyBlender":
        with open(cfg_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        pol_cfgs: List[PolicyConfig] = []
        for p in raw.get("policies", []):
            pol_cfgs.append(
                PolicyConfig(
                    name=str(p["name"]),
                    base_weight=float(p.get("base_weight", 0.0)),
                )
            )

        b = raw.get("blend", {}) or {}
        blend_cfg = BlendConfig(
            rl_confidence_boost=float(b.get("rl_confidence_boost", 0.5)),
            high_vol_boost_momentum=float(b.get("high_vol_boost_momentum", 0.2)),
            low_vol_boost_trend=float(b.get("low_vol_boost_trend", 0.2)),
            min_total_weight=float(b.get("min_total_weight", 0.001)),
        )

        return cls(policies=policies, policy_cfgs=pol_cfgs, blend_cfg=blend_cfg)

    # ------------------------------------------------------------------ #

    def _compute_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        vol = float(context.get("volatility", 1.0))
        rl_conf = float(context.get("rl_confidence", 0.5))

        weights: Dict[str, float] = {}

        for pc in self.policy_cfgs:
            weights[pc.name] = pc.base_weight

        # RL confidence tilt
        if "rl_equity" in weights:
            boost = self.blend_cfg.rl_confidence_boost * rl_conf
            weights["rl_equity"] += boost

        # Volatility regime
        if vol >= 1.5:
            if "momentum" in weights:
                weights["momentum"] += self.blend_cfg.high_vol_boost_momentum
        elif vol <= 0.7:
            if "trend" in weights:
                weights["trend"] += self.blend_cfg.low_vol_boost_trend

        total = sum(max(0.0, w) for w in weights.values())
        if total < self.blend_cfg.min_total_weight:
            logger.warning("Total weight too small (%.6f), using uniform", total)
            n = max(1, len(weights))
            return {k: 1.0 / n for k in weights}

        return {k: max(0.0, w) / total for k, w in weights.items()}

    def blend(self, obs: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
            {
              "action": blended_action,
              "weights": {policy_name: weight},
              "raw_actions": {policy_name: raw_action}
            }
        """
        weights = self._compute_weights(context)

        raw_actions: Dict[str, Any] = {}
        blended = None
        first = True

        for name, w in weights.items():
            pol = self.policies.get(name)
            if pol is None:
                logger.warning("Policy '%s' not present – skipping in blend", name)
                continue

            a = pol.predict(obs)
            raw_actions[name] = a

            if first:
                blended = a * w  # type: ignore
                first = False
            else:
                blended = blended + a * w  # type: ignore

        return {
            "action": blended,
            "weights": weights,
            "raw_actions": raw_actions,
        }
