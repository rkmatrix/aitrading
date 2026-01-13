# ai/fusion/fusion_engine.py
# Phase 92 + 123 — FusionEngine wrapper around FusionBrain

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class FusionEngine:
    """
    FusionEngine wraps FusionBrain and applies Phase 123 adjustments:

        • MetaStabilityEngine (clamp_factor, freeze decisions)
        • ExecutionQualityMemory (aggression_scale)

    It mirrors FusionBrain.fuse(...) but post-processes the output.
    """

    def __init__(
        self,
        *,
        fusion_brain: Any,
        meta_engine: Optional[Any] = None,
        eqm: Optional[Any] = None,
    ) -> None:
        self._fusion_brain = fusion_brain
        self._meta_engine = meta_engine
        self._eqm = eqm

    def attach_stability(self, meta_engine: Any) -> None:
        self._meta_engine = meta_engine
        log.info("FusionEngine: Meta-Stability engine attached.")

    def attach_eqm(self, eqm: Any) -> None:
        self._eqm = eqm
        log.info("FusionEngine: Execution Quality Memory attached.")

    # ---------------------------------------------------------
    # Core fuse()
    # ---------------------------------------------------------
    def fuse(self, **kwargs: Any) -> Dict[str, Any]:
        fused = self._fusion_brain.fuse(**kwargs)

        if not isinstance(fused, dict):
            return fused

        return self._post_process(fused)

    # ---------------------------------------------------------
    # Apply meta-stability + EQM scaling
    # ---------------------------------------------------------
    def _post_process(self, fused: Dict[str, Any]) -> Dict[str, Any]:
        fused = dict(fused)
        score = float(fused.get("fused_score", 0.0))
        weights = fused.get("weights", None)

        # ---- 1) Meta-Stability clamp ----
        if self._meta_engine is not None:
            try:
                st = self._meta_engine.compute()
                clamp = float(st.get("clamp_factor", 1.0))
                freeze = bool(st.get("decision_freeze", False))

                if freeze:
                    log.warning("FusionEngine: Meta-stability freeze → forcing fused_score=0.")
                    fused["fused_score"] = 0.0
                    if isinstance(weights, dict):
                        fused["weights"] = {k: 0.0 for k in weights}
                    return fused

                score *= clamp
            except Exception as e:
                log.exception("FusionEngine: meta_engine.compute failed: %s", e)

        # ---- 2) EQM aggression scaling ----
        if self._eqm is not None:
            try:
                q = self._eqm.compute()
                scale = float(q.get("aggression_scale", 1.0))
                score *= scale
            except Exception as e:
                log.exception("FusionEngine: eqm.compute failed: %s", e)

        fused["fused_score"] = score

        if isinstance(weights, dict):
            fused["weights"] = {k: float(v) for k, v in weights.items()}

        return fused

    # ---------------------------------------------------------
    # Delegate missing attributes to underlying FusionBrain
    # ---------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        try:
            return getattr(self._fusion_brain, item)
        except AttributeError as e:
            raise AttributeError(item) from e
