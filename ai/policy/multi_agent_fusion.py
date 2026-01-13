"""
ai/policy/multi_agent_fusion.py
--------------------------------
Phase 92 – Multi-Agent Fusion Engine

Blends multiple agents:
    - ML AlphaZoo agent
    - Execution PPO agent
    - Momentum agent
    - Regime/Risk agent

Each agent exposes:
    decide(obs, portfolio=None, info=None) -> dict:
        {
            "action": <np.ndarray | list | dict | str | int>,
            "confidence": float,   # 0..1
            "risk_score": float,   # 0..1 (1 = risky)
            "meta": dict
        }

MultiAgentFusionEngine:
    fused = engine.decide(obs, portfolio=..., info=...)
    fused["action"] -> blended action
    fused["fusion"] -> diagnostics (weights, context, per-agent outputs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("MultiAgentFusionEngine")


@dataclass
class AgentWrapper:
    name: str
    kind: str              # "ml", "execution", "momentum", "regime_risk", ...
    base_weight: float
    obj: Any               # underlying agent with decide(...)


class MultiAgentFusionEngine:
    """
    Phase 92 – Multi-Agent Fusion Engine.

    Responsibilities:
    -----------------
    - Call all enabled agents.
    - Compute dynamic weights based on:
        * volatility
        * drawdown
        * trend strength
        * ML variance / uncertainty
    - Blend per-agent actions into a single final action.
    - Emit diagnostics for logging / dashboards.
    """

    def __init__(
        self,
        agents: List[AgentWrapper],
        fusion_cfg: Dict[str, Any],
        *,
        broker: Any | None = None,
        pbrain: Any | None = None,
        signal_engine: Any | None = None,
    ) -> None:
        self.agents = agents
        self.cfg = fusion_cfg or {}

        # Optional context providers (not required, but useful for advanced fusion)
        self.broker = broker
        self.pbrain = pbrain
        self.signal_engine = signal_engine

        self.vol_sensitivity = float(self.cfg.get("vol_sensitivity", 0.6))
        self.dd_sensitivity = float(self.cfg.get("dd_sensitivity", 0.10))
        self.trend_boost_threshold = float(self.cfg.get("trend_boost_threshold", 0.30))
        self.ml_var_high = float(self.cfg.get("ml_var_high", 0.40))
        self.min_weight = float(self.cfg.get("min_weight", 0.02))
        self.max_exposure_multiplier = float(
            self.cfg.get("max_exposure_multiplier", 1.5)
        )

        logger.info(
            "MultiAgentFusionEngine initialized with %d agents",
            len(self.agents),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(
        self,
        obs: Dict[str, Any],
        *,
        portfolio: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Returns:
            dict:
                {
                    "action": ...,
                    "fusion": {
                        "weights": {agent_name: weight, ...},
                        "confidences": {agent_name: conf, ...},
                        "risk_scores": {agent_name: risk, ...},
                        "context": {...},
                        "raw_actions": {agent_name: ...},
                    }
                }
        """
        if info is None:
            info = {}

        # --- 1) Call all agents ----------------------------------------
        per_agent = {}
        for aw in self.agents:
            try:
                out = aw.obj.decide(obs, portfolio=portfolio, info=info)
                action = out.get("action")
                conf = float(out.get("confidence", 1.0))
                risk = float(out.get("risk_score", 0.0))
                meta = out.get("meta", {})

                per_agent[aw.name] = {
                    "kind": aw.kind,
                    "base_weight": aw.base_weight,
                    "action": action,
                    "confidence": conf,
                    "risk_score": risk,
                    "meta": meta,
                }
            except Exception as e:  # noqa: BLE001
                logger.exception("Agent %s failed in decide(): %s", aw.name, e)
                # fallback: zero-confidence no-op
                per_agent[aw.name] = {
                    "kind": aw.kind,
                    "base_weight": aw.base_weight,
                    "action": None,
                    "confidence": 0.0,
                    "risk_score": 1.0,
                    "meta": {"error": str(e)},
                }

        # --- 2) Compute dynamic weights --------------------------------
        context = self._extract_context(obs=obs, portfolio=portfolio, info=info)
        weights = self._compute_dynamic_weights(per_agent, context)

        # --- 3) Blend actions ------------------------------------------
        fused_action = self._blend_actions(per_agent, weights)

        return {
            "action": fused_action,
            "fusion": {
                "weights": weights,
                "confidences": {k: v["confidence"] for k, v in per_agent.items()},
                "risk_scores": {k: v["risk_score"] for k, v in per_agent.items()},
                "context": context,
                "raw_actions": {k: v["action"] for k, v in per_agent.items()},
                "per_agent_meta": {k: v["meta"] for k, v in per_agent.items()},
            },
        }

    # ------------------------------------------------------------------
    # Context + weights
    # ------------------------------------------------------------------
    def _extract_context(
        self,
        *,
        obs: Dict[str, Any],
        portfolio: Optional[Dict[str, Any]],
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extract normalized context values for weight adjustment.

        Expected to be 0..1 or -1..1 scaled where applicable.
        """
        # try multiple sources (obs, portfolio, info) to be flexible
        # you can refine this once wired into your env.
        vol = (
            info.get("realized_volatility")
            or obs.get("realized_volatility")
            or info.get("volatility")
            or 0.2
        )
        drawdown = (
            info.get("drawdown")
            or obs.get("drawdown")
            or (portfolio or {}).get("drawdown")
            or 0.0
        )
        trend_strength = (
            info.get("trend_strength")
            or obs.get("trend_strength")
            or 0.0  # -1..1
        )
        ml_variance = (
            info.get("ml_variance")
            or obs.get("ml_variance")
            or 0.0
        )

        return {
            "volatility": float(vol),
            "drawdown": float(drawdown),
            "trend_strength": float(trend_strength),
            "ml_variance": float(ml_variance),
        }

    def _compute_dynamic_weights(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        ctx: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply dynamic re-weighting based on context and agent meta.
        """
        names = list(per_agent.keys())
        base = np.array([per_agent[n]["base_weight"] for n in names], dtype=float)

        # guarantee non-negative
        base = np.clip(base, 0.0, None)

        vol = ctx.get("volatility", 0.2)
        dd = ctx.get("drawdown", 0.0)
        trend = ctx.get("trend_strength", 0.0)
        ml_var = ctx.get("ml_variance", 0.0)

        # --- Start from base weights -----------------------------------
        w = base.copy()

        # --- 2b) Volatility high → defensive (regime_risk & execution) --
        if vol > self.vol_sensitivity:
            for i, name in enumerate(names):
                kind = per_agent[name]["kind"]
                if kind in ("regime_risk", "execution"):
                    w[i] *= 1.3
                elif kind in ("momentum", "ml"):
                    w[i] *= 0.7

        # --- 2c) Drawdown high → defensive -----------------------------
        if dd > self.dd_sensitivity:
            for i, name in enumerate(names):
                kind = per_agent[name]["kind"]
                if kind in ("regime_risk", "execution"):
                    w[i] *= 1.3
                elif kind == "momentum":
                    w[i] *= 0.6
                elif kind == "ml":
                    w[i] *= 0.8

        # --- 2d) Trend boost -------------------------------------------
        if abs(trend) > self.trend_boost_threshold:
            for i, name in enumerate(names):
                kind = per_agent[name]["kind"]
                if kind == "momentum":
                    w[i] *= 1.5
                elif kind == "regime_risk":
                    w[i] *= 0.9

        # --- 2e) ML variance: uncertainty penalty ----------------------
        if ml_var > self.ml_var_high:
            for i, name in enumerate(names):
                kind = per_agent[name]["kind"]
                if kind == "ml":
                    w[i] *= 0.5

        # --- 2f) Agent-level confidence & risk -------------------------
        for i, name in enumerate(names):
            conf = float(per_agent[name]["confidence"])
            risk = float(per_agent[name]["risk_score"])

            # more confidence → higher weight
            w[i] *= np.clip(conf, 0.0, 1.0)

            # higher risk → slight penalty (optional)
            w[i] *= (1.0 - 0.3 * np.clip(risk, 0.0, 1.0))

        # --- Normalize & enforce minimum -------------------------------
        w_sum = float(w.sum())
        if w_sum <= 0:
            # fallback: uniform
            w = np.ones_like(w) / len(w)
        else:
            w /= w_sum

        # enforce minimum for enabled agents to avoid complete collapse
        if self.min_weight > 0:
            w = np.maximum(w, self.min_weight)
            w /= float(w.sum())

        weights = {name: float(w[i]) for i, name in enumerate(names)}
        logger.debug("Fusion weights: %s | ctx=%s", weights, ctx)
        return weights

    # ------------------------------------------------------------------
    # Action fusion
    # ------------------------------------------------------------------
    def _blend_actions(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Any:
        """
        Blend actions of possibly different types.

        Supported patterns:
            - dict[symbol -> float]
            - list/np.ndarray of floats
            - scalar (float or int)
            - discrete label (str/int):
                -> choose agent with max weight*confidence
        """
        # inspect first non-None action to decide mode
        example = None
        for v in per_agent.values():
            if v["action"] is not None:
                example = v["action"]
                break

        if example is None:
            # no-op
            return None

        if isinstance(example, dict):
            return self._blend_dict_actions(per_agent, weights)
        if isinstance(example, (list, tuple, np.ndarray)):
            return self._blend_vector_actions(per_agent, weights)
        if isinstance(example, (float, int)):
            return self._blend_scalar_actions(per_agent, weights)
        # fallback: treat as label
        return self._select_label_action(per_agent, weights)

    def _blend_dict_actions(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        symbols = set()
        for v in per_agent.values():
            act = v["action"]
            if isinstance(act, dict):
                symbols |= set(act.keys())

        fused: Dict[str, float] = {s: 0.0 for s in symbols}
        for name, v in per_agent.items():
            act = v["action"]
            if not isinstance(act, dict):
                continue
            w = float(weights.get(name, 0.0))
            for sym, val in act.items():
                fused[sym] += w * float(val)

        return fused

    def _blend_vector_actions(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> List[float]:
        # assume same length vectors
        fused_vec: Optional[np.ndarray] = None
        for name, v in per_agent.items():
            act = v["action"]
            if act is None:
                continue
            arr = np.asarray(act, dtype=float)
            w = float(weights.get(name, 0.0))
            if fused_vec is None:
                fused_vec = np.zeros_like(arr)
            fused_vec += w * arr

        return fused_vec.tolist() if fused_vec is not None else []

    def _blend_scalar_actions(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> float:
        fused = 0.0
        for name, v in per_agent.items():
            act = v["action"]
            if act is None:
                continue
            w = float(weights.get(name, 0.0))
            fused += w * float(act)
        return float(fused)

    def _select_label_action(
        self,
        per_agent: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Any:
        """
        For discrete labels (e.g., actions "BUY", "SELL", "HOLD"),
        pick the label from the agent with the strongest weight*confidence.
        """
        best_name = None
        best_score = -1.0
        best_action = None
        for name, v in per_agent.items():
            act = v["action"]
            if act is None:
                continue
            conf = float(v["confidence"])
            w = float(weights.get(name, 0.0))
            score = conf * w
            if score > best_score:
                best_score = score
                best_name = name
                best_action = act

        logger.debug("Label fusion winner: %s (score=%.4f)", best_name, best_score)
        return best_action
