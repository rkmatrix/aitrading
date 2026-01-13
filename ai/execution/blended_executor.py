# ai/execution/blended_executor.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ai.supervisor.policy_supervisor import MultiPolicySupervisor

def _safe(arr):
    return np.array(arr, dtype=float)

class BlendedExecutor:
    """
    Combines multiple policy predictions using the Supervisor's weights
    and each policy's confidence.

    Default strategy: weighted average of target positions.
    weight for a policy i = W_i * (confidence_i ** confidence_power)
    """

    def __init__(self, supervisor: MultiPolicySupervisor):
        self.sup = supervisor

    def blend(self, prediction_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            prediction_pack: result from supervisor.step() or supervisor.predict_all()
                If using supervisor.step(), pass prediction_pack["predictions"] here.

        Returns:
            {
              "targets": {symbol: blended_target [-1..1], ...},
              "confidence": float [0..1],
              "meta": {"per_policy": {...}, "weights": {...}}
            }
        """
        # Accept either {"policy_name": {pred...}} or the full supervisor.step() result
        preds = prediction_pack
        if "predictions" in prediction_pack and isinstance(prediction_pack["predictions"], dict):
            preds = prediction_pack["predictions"]

        weights = self.sup.get_weights()
        conf_pow = self.sup.cfg.confidence_power

        symbols = self.sup.cfg.symbols
        agg_num = {s: 0.0 for s in symbols}
        agg_den = {s: 0.0 for s in symbols}
        per_policy = {}

        for name, pred in preds.items():
            w = float(weights.get(name, 0.0))
            c = float(pred.get("confidence", 1.0))
            eff = w * (c ** conf_pow)
            tmap = pred.get("targets", {}) or {}
            per_policy[name] = {"weight": w, "confidence": c, "effective_weight": eff, "targets": tmap}

            for sym in symbols:
                target = float(tmap.get(sym, 0.0))
                agg_num[sym] += eff * target
                agg_den[sym] += eff

        blended_targets = {}
        for sym in symbols:
            if agg_den[sym] <= 1e-12:
                blended_targets[sym] = 0.0
            else:
                blended_targets[sym] = float(np.clip(agg_num[sym] / agg_den[sym], -1.0, 1.0))

        # Overall blend confidence: normalized sum of effective weights
        total_eff = float(sum(p["effective_weight"] for p in per_policy.values()))
        overall_conf = 0.0 if total_eff <= 1e-12 else float(np.clip(total_eff, 0.0, 1.0))

        return {
            "targets": blended_targets,
            "confidence": overall_conf,
            "meta": {"per_policy": per_policy, "weights": weights},
        }
