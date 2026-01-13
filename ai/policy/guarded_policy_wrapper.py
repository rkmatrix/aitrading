"""
ai/policy/guarded_policy_wrapper.py
-----------------------------------

GuardedPolicyWrapper (Phase 122.3)

Wraps an underlying policy and applies GuardrailRuntimeHandler to its
actions before they reach the rest of the pipeline.

Assumes the underlying policy exposes:

    - act(obs) -> action_dict

Where action_dict may include:
    - "weights": dict symbol -> target weight
    - or any structure; we treat it generically and extract a few metrics.

You can extend _extract_action_metrics / _apply_clamps to match your
specific policy schema.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional

from ai.safety.auto_guardrails import GuardrailRuntimeHandler, GuardrailDecision

logger = logging.getLogger(__name__)


class GuardedPolicyWrapper:
    """
    Wraps a policy with runtime guardrails.
    """

    def __init__(
        self,
        base_policy: Any,
        guardrails: Optional[GuardrailRuntimeHandler] = None,
        logger_name: str = "GuardedPolicy",
    ) -> None:
        self.base_policy = base_policy
        self.guardrails = guardrails
        self.log = logging.getLogger(logger_name)

    # ------------------------------------------------------------------ #
    # Public API (mirrors base policy)
    # ------------------------------------------------------------------ #

    def act(self, obs: Any) -> Dict[str, Any]:
        """
        Get action from base_policy, then apply guardrails.

        If guardrails block:
            - we return a "no-op" / flat action.

        If guardrails clamp:
            - we optionally scale down weights.
        """
        action = self.base_policy.act(obs) if hasattr(self.base_policy, "act") else self.base_policy(obs)  # type: ignore[misc]

        if self.guardrails is None:
            return action

        metrics = self._extract_action_metrics(action)
        decision: GuardrailDecision = self.guardrails.check(
            event="act",
            metrics=metrics,
            context={"actor": type(self.base_policy).__name__},
        )

        if decision.is_blocked:
            self.log.error(
                "⛔ Guardrails blocked policy action (metrics=%s): %s",
                metrics,
                "; ".join(decision.reasons),
            )
            return self._flat_action_like(action)

        if decision.is_clamp:
            self.log.warning(
                "Guardrails clamped policy action; metrics=%s",
                decision.metrics,
            )
            return self._apply_clamps(action, decision.metrics)

        return action

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _extract_action_metrics(self, action: Dict[str, Any]) -> Dict[str, float]:
        """
        Derive a few generic metrics from an action dict:

        - max_abs_weight_change: if action contains 'weights' and maybe 'prev_weights'
        - num_active: number of non-zero weights
        - gross_abs_weight: sum abs weights
        """
        metrics: Dict[str, float] = {}

        weights = action.get("weights") if isinstance(action, dict) else None
        prev_weights = action.get("prev_weights") if isinstance(action, dict) else None

        if isinstance(weights, dict):
            gross = sum(abs(float(w)) for w in weights.values())
            num_active = sum(1 for w in weights.values() if abs(float(w)) > 1e-6)
            metrics["gross_abs_weight"] = gross
            metrics["num_active_symbols"] = float(num_active)

            if isinstance(prev_weights, dict):
                max_delta = 0.0
                for sym, w in weights.items():
                    prev = float(prev_weights.get(sym, 0.0))
                    delta = abs(float(w) - prev)
                    max_delta = max(max_delta, delta)
                metrics["max_abs_weight_change"] = max_delta

        return metrics

    def _flat_action_like(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a "no-op" / flat action using the same structure.
        """
        if not isinstance(action, dict):
            return {"weights": {}}

        out = dict(action)
        if isinstance(out.get("weights"), dict):
            out["weights"] = {sym: 0.0 for sym in out["weights"].keys()}
        return out

    def _apply_clamps(self, action: Dict[str, Any], clamp_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally scale down weights based on a factor from clamp_metrics,
        e.g. clamp_metrics["scale"] or similar.
        """
        if not isinstance(action, dict):
            return action

        scale = float(clamp_metrics.get("scale", 1.0))
        if math.isclose(scale, 1.0):
            return action

        out = dict(action)
        weights = out.get("weights")
        if isinstance(weights, dict):
            out["weights"] = {sym: float(w) * scale for sym, w in weights.items()}
        return out
