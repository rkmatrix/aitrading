# ai/evolution/weight_evolution_manager.py
"""
Phase 66 – Weight Evolution Manager

Coordinates:
    • AgentPerformanceTracker
    • Weight evolvers (Heuristic, RL-style bandit, Genetic)
    • Reward computation based on decisions

Reward modes supported:
    - "directional":   action vs price_drift sign
    - "pnl":           uses pnl from order_meta / context
    - "confidence":    pure confidence-based reward
    - "hybrid":        0.7 * directional + 0.3 * pnl

Evolver modes supported:
    - "heuristic"      → HeuristicWeightEvolver (A)
    - "rl"             → RLWeightEvolver (B)
    - "genetic"        → GeneticWeightEvolver (C)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import logging
import math

from ai.agents.base_agent import AgentContext
from ai.agents.votes import FusedDecision
from ai.agents.multi_agent_brain import MultiAgentBrain
from ai.execution.execution_pipeline import ExecutionResult
from ai.evolution.agent_performance_tracker import AgentPerformanceTracker, AgentStats
from ai.evolution.agent_weight_evolver import (
    BaseWeightEvolver,
    HeuristicWeightEvolver,
    RLWeightEvolver,
    GeneticWeightEvolver,
)

logger = logging.getLogger(__name__)


@dataclass
class WeightEvolutionConfig:
    enabled: bool = True
    mode: str = "heuristic"          # "heuristic" | "rl" | "genetic"
    reward_mode: str = "directional" # "directional" | "pnl" | "confidence" | "hybrid"
    update_interval: int = 20        # decisions between weight updates
    min_samples: int = 10            # minimum total updates before evolution
    performance_path: str = "data/runtime/phase65_agent_performance.json"
    genetic_state_path: str = "data/runtime/phase65_population.json"
    reward_scale_pnl: float = 100.0  # divisor for pnl-based reward


class WeightEvolutionManager:
    """
    Encapsulates online weight evolution logic for MultiAgentBrain.

    Typical usage in realtime loop:

        config = cfg.get("evolution", {})
        manager = WeightEvolutionManager(brain, config)

        for each decision:
            manager.on_decision(ctx, decision, result)
    """

    def __init__(self, brain: MultiAgentBrain, config: Dict[str, Any]) -> None:
        self.brain = brain
        self.cfg = self._parse_config(config)

        self.enabled = self.cfg.enabled
        self.reward_mode = self.cfg.reward_mode.lower()
        self.update_interval = int(self.cfg.update_interval)
        self.min_samples = int(self.cfg.min_samples)

        # Performance tracker
        perf_path = Path(self.cfg.performance_path)
        self.tracker = AgentPerformanceTracker.load_json(perf_path)

        # Evolver
        mode = self.cfg.mode.lower()
        if mode == "heuristic":
            self.evolver: BaseWeightEvolver = HeuristicWeightEvolver()
        elif mode == "rl":
            self.evolver = RLWeightEvolver()
        elif mode == "genetic":
            self.evolver = GeneticWeightEvolver(
                state_path=self.cfg.genetic_state_path
            )
        else:
            logger.warning("Unknown evolution mode '%s', defaulting to heuristic.", mode)
            self.evolver = HeuristicWeightEvolver()

        self._perf_path = perf_path
        self._decision_counter = 0
        self._total_updates = sum(s.count for s in self.tracker.get_all_stats().values())

        logger.info(
            "WeightEvolutionManager initialized: enabled=%s mode=%s reward_mode=%s",
            self.enabled,
            mode,
            self.reward_mode,
        )

    # ------------------------------------------------------------------ #
    # Config parsing
    # ------------------------------------------------------------------ #

    def _parse_config(self, cfg: Dict[str, Any]) -> WeightEvolutionConfig:
        return WeightEvolutionConfig(
            enabled=bool(cfg.get("enabled", True)),
            mode=str(cfg.get("mode", "heuristic")),
            reward_mode=str(cfg.get("reward_mode", "directional")),
            update_interval=int(cfg.get("update_interval", 20)),
            min_samples=int(cfg.get("min_samples", 10)),
            performance_path=str(cfg.get("performance_path", "data/runtime/phase65_agent_performance.json")),
            genetic_state_path=str(cfg.get("genetic_state_path", "data/runtime/phase65_population.json")),
            reward_scale_pnl=float(cfg.get("reward_scale_pnl", 100.0)),
        )

    # ------------------------------------------------------------------ #
    # Public hook
    # ------------------------------------------------------------------ #

    def on_decision(
        self,
        ctx: AgentContext,
        decision: FusedDecision,
        result: ExecutionResult,
    ) -> None:
        """
        Called once per decision in the realtime loop.

        Steps:
            1. Compute per-agent rewards based on reward_mode.
            2. Update performance tracker.
            3. Periodically evolve and apply weights to MultiAgentBrain.
        """
        if not self.enabled:
            return

        rewards = self._compute_rewards(ctx, decision, result)
        if rewards:
            self.tracker.bulk_update(rewards)
            self._total_updates += len(rewards)
            self._save_performance()

        self._decision_counter += 1
        if (
            self._decision_counter >= self.update_interval
            and self._total_updates >= self.min_samples
        ):
            self._decision_counter = 0
            self._evolve_weights()

    # ------------------------------------------------------------------ #
    # Reward computation
    # ------------------------------------------------------------------ #

    def _compute_rewards(
        self,
        ctx: AgentContext,
        decision: FusedDecision,
        result: ExecutionResult,
    ) -> Dict[str, float]:
        if not decision.votes:
            return {}

        mode = self.reward_mode
        rewards: Dict[str, float] = {}

        for v in decision.votes:
            if mode == "directional":
                r = self._reward_directional(ctx, v.action, v.confidence)
            elif mode == "pnl":
                r = self._reward_pnl(ctx, result, v.confidence)
            elif mode == "confidence":
                r = self._reward_confidence(v.action, v.confidence)
            elif mode == "hybrid":
                rd = self._reward_directional(ctx, v.action, v.confidence)
                rp = self._reward_pnl(ctx, result, v.confidence)
                r = 0.7 * rd + 0.3 * rp
            else:
                r = 0.0

            if r != 0.0:
                rewards[v.agent_id] = rewards.get(v.agent_id, 0.0) + r

        if rewards:
            logger.debug("WeightEvolutionManager rewards: %s", rewards)
        return rewards

    def _reward_directional(
        self,
        ctx: AgentContext,
        action: str,
        confidence: float,
    ) -> float:
        """
        Uses ctx.extra['price_drift'] as a proxy for direction.

        Intuition:
            - BUY is rewarded if drift > 0
            - SELL is rewarded if drift < 0
            - HOLD → 0 reward
        """
        drift = float(ctx.extra.get("price_drift", 0.0))
        if abs(drift) < 1e-6:
            return 0.0

        if action == "HOLD":
            return 0.0

        sign_drift = 1.0 if drift > 0 else -1.0
        sign_act = 1.0 if action == "BUY" else -1.0
        reward = sign_drift * sign_act * max(0.1, min(1.0, confidence))
        return reward

    def _reward_pnl(
        self,
        ctx: AgentContext,
        result: ExecutionResult,
        confidence: float,
    ) -> float:
        """
        PnL-based reward.

        Attempts to read pnl from:
            - result.order_meta["pnl"]
            - ctx.extra["realized_pnl"]
        """
        scale = self.cfg.reward_scale_pnl or 1.0
        pnl = 0.0
        if result.order_meta:
            pnl = float(result.order_meta.get("pnl", 0.0) or 0.0)
        if abs(pnl) < 1e-9:
            pnl = float(ctx.extra.get("realized_pnl", 0.0) or 0.0)

        if abs(pnl) < 1e-9:
            return 0.0

        reward = (pnl / scale) * max(0.1, min(1.0, confidence))
        return reward

    def _reward_confidence(self, action: str, confidence: float) -> float:
        """
        Pure confidence-based reward:
            - Positive reward for non-HOLD actions proportional to confidence.
            - HOLD gets a small reward if confident, negative if over-confident.
        """
        c = max(0.0, min(1.0, confidence))
        if action in ("BUY", "SELL"):
            return c
        # Penalize excessive HOLD confidence slightly
        return 0.2 * (c - 0.5)

    # ------------------------------------------------------------------ #
    # Evolution
    # ------------------------------------------------------------------ #

    def _evolve_weights(self) -> None:
        current = self.brain.get_agent_weights()
        stats = self.tracker.get_all_stats()
        if not current:
            return

        new_weights = self.evolver.evolve(current, stats)
        # sanity clamp: avoid NaNs
        sane_weights: Dict[str, float] = {}
        for aid, w in new_weights.items():
            if not math.isfinite(w):
                sane_weights[aid] = current.get(aid, 1.0)
            else:
                sane_weights[aid] = w

        self.brain.set_agent_weights(sane_weights)
        logger.info("WeightEvolutionManager applied new weights: %s", sane_weights)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save_performance(self) -> None:
        try:
            self.tracker.save_json(self._perf_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to save agent performance to %s: %s", self._perf_path, exc, exc_info=True)


__all__ = ["WeightEvolutionManager", "WeightEvolutionConfig"]
