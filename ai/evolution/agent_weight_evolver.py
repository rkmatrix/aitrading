# ai/evolution/agent_weight_evolver.py
"""
Phase 65 – Agent Weight Evolvers (A, B, C)

Provides multiple strategies to evolve agent weights based on performance stats:

A) HeuristicWeightEvolver
    - Increment/decrement weights proportional to mean reward
    - Simple, stable, production-friendly

B) RLWeightEvolver (multi-armed bandit style)
    - Treats each agent as an "arm"
    - Uses softmax over mean reward to derive weights
    - Temperature parameter controls exploration vs exploitation

C) GeneticWeightEvolver
    - Maintains a population of candidate weight vectors
    - Evolves them via selection, crossover, mutation
    - Intended for offline / batch experimentation

All evolvers share a common interface:

    evolve(current_weights: Dict[str, float],
           stats: Dict[str, AgentStats]) -> Dict[str, float]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import math
import random
import logging

from .agent_performance_tracker import AgentStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseWeightEvolver:
    """
    Abstract interface for agent weight evolvers.
    """

    def evolve(
        self,
        current_weights: Dict[str, float],
        stats: Dict[str, AgentStats],
    ) -> Dict[str, float]:
        """
        Given current weights and agent stats, return new weights.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# A) HeuristicWeightEvolver
# ---------------------------------------------------------------------------


@dataclass
class HeuristicWeightEvolver(BaseWeightEvolver):
    """
    Simple heuristic-based evolver.

    Rules:
        - Normalize mean_reward into [-1, 1] by a scale factor.
        - If > 0: increase weight by lr_up * score
        - If < 0: decrease weight by lr_down * abs(score)
        - Clip between [min_weight, max_weight]
        - Optional global decay to gently pull weights to baseline.

    Config:
        lr_up:       learning rate for positive performance
        lr_down:     learning rate for negative performance
        min_weight:  lower clip
        max_weight:  upper clip
        decay:       multiplicative decay per update (e.g. 0.01 → 1% toward baseline)
        baseline:    baseline weight to decay toward (usually 1.0)
        reward_scale:divisor to normalize mean_reward
    """

    lr_up: float = 0.2
    lr_down: float = 0.2
    min_weight: float = 0.1
    max_weight: float = 3.0
    decay: float = 0.01
    baseline: float = 1.0
    reward_scale: float = 1.0

    def evolve(
        self,
        current_weights: Dict[str, float],
        stats: Dict[str, AgentStats],
    ) -> Dict[str, float]:
        new_weights: Dict[str, float] = dict(current_weights)

        for aid, w in current_weights.items():
            s = stats.get(aid, AgentStats(agent_id=aid))
            mean_r = s.mean_reward
            # normalize reward into a convenient range
            score = mean_r / self.reward_scale if self.reward_scale > 0 else mean_r
            # clamp score into [-1, 1]
            score = max(-1.0, min(1.0, score))

            # apply decay toward baseline
            w = w + self.decay * (self.baseline - w)

            if score > 0:
                delta = self.lr_up * score
                w = w + delta
            elif score < 0:
                delta = self.lr_down * abs(score)
                w = w - delta

            w = max(self.min_weight, min(self.max_weight, w))
            new_weights[aid] = w

        logger.info("HeuristicWeightEvolver updated weights: %s", new_weights)
        return new_weights


# ---------------------------------------------------------------------------
# B) RLWeightEvolver – softmax bandit
# ---------------------------------------------------------------------------


@dataclass
class RLWeightEvolver(BaseWeightEvolver):
    """
    RL-style evolver using a softmax multi-armed bandit approach.

    Idea:
        - Each agent has an estimated value (here: its mean_reward).
        - Convert these values into a probability distribution via softmax.
        - Map probabilities into weights by scaling to [min_weight, max_weight].

    Config:
        temperature:  higher → more uniform weights; lower → more peaked
        min_weight, max_weight: output range for weights
        eps: small constant to avoid division by zero
    """

    temperature: float = 0.5
    min_weight: float = 0.3
    max_weight: float = 3.0
    eps: float = 1e-8

    def evolve(
        self,
        current_weights: Dict[str, float],
        stats: Dict[str, AgentStats],
    ) -> Dict[str, float]:
        agent_ids = list(current_weights.keys())
        if not agent_ids:
            return current_weights

        # Gather "values" for each agent (mean rewards)
        values = []
        for aid in agent_ids:
            s = stats.get(aid, AgentStats(agent_id=aid))
            values.append(s.mean_reward)

        # Softmax over values / temperature
        max_v = max(values) if values else 0.0
        scaled = [
            math.exp((v - max_v) / max(self.temperature, self.eps)) for v in values
        ]
        total = sum(scaled) or self.eps
        probs = [x / total for x in scaled]

        # Map probabilities into weights in [min_weight, max_weight]
        span = self.max_weight - self.min_weight
        new_weights: Dict[str, float] = {}
        for aid, p in zip(agent_ids, probs):
            w = self.min_weight + span * p
            new_weights[aid] = w

        logger.info("RLWeightEvolver updated weights (softmax bandit): %s", new_weights)
        return new_weights


# ---------------------------------------------------------------------------
# C) GeneticWeightEvolver – population-based
# ---------------------------------------------------------------------------


@dataclass
class Genome:
    """
    Represents one candidate set of weights.

    Fields:
        genes:    mapping agent_id -> weight
        fitness:  scalar fitness value (higher is better)
    """

    genes: Dict[str, float]
    fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"genes": dict(self.genes), "fitness": self.fitness}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(genes=dict(d.get("genes", {})), fitness=float(d.get("fitness", 0.0)))


@dataclass
class GeneticWeightEvolver(BaseWeightEvolver):
    """
    Genetic algorithm for evolving agent weight vectors.

    Intended usage:
        - Offline or batch mode, evaluating each genome over some backtest.
        - You provide fitness values (e.g. PnL, Sharpe) externally.

    Key operations:
        - select_top: keep top K genomes by fitness
        - crossover: randomly mix genes between parents
        - mutate: random noise applied to weights

    Methods:
        - initialize_population(agent_ids, base_weights)
        - step(fitness_map) -> updates population
        - best_genome() -> returns the best genes
        - evolve(current_weights, stats) -> returns best or current + small tweak
    """

    population_size: int = 20
    elite_count: int = 5
    mutation_rate: float = 0.2
    mutation_scale: float = 0.3
    min_weight: float = 0.1
    max_weight: float = 3.0
    state_path: Optional[str] = None  # optional JSON storage for population

    def __post_init__(self) -> None:
        self._population: List[Genome] = []

        if self.state_path:
            self._load_state(Path(self.state_path))

    # ---- Population management -------------------------------------------

    def initialize_population(
        self,
        agent_ids: List[str],
        base_weights: Dict[str, float],
    ) -> None:
        if self._population:
            return  # already initialized

        for _ in range(self.population_size):
            genes = {}
            for aid in agent_ids:
                base = base_weights.get(aid, 1.0)
                noise = random.uniform(-0.5, 0.5)
                w = max(self.min_weight, min(self.max_weight, base + noise))
                genes[aid] = w
            self._population.append(Genome(genes=genes, fitness=0.0))

        self._save_state_if_needed()

    def best_genome(self) -> Optional[Genome]:
        if not self._population:
            return None
        return max(self._population, key=lambda g: g.fitness)

    def step(self, fitness_map: Dict[int, float]) -> None:
        """
        Perform one evolutionary step using a dict:
            genome_index -> fitness_value
        """
        if not self._population:
            return

        # Update fitness values
        for idx, fit in fitness_map.items():
            if 0 <= idx < len(self._population):
                self._population[idx].fitness = float(fit)

        # Selection
        self._population.sort(key=lambda g: g.fitness, reverse=True)
        elites = self._population[: self.elite_count]

        # Reproduction
        new_pop: List[Genome] = list(elites)
        while len(new_pop) < self.population_size:
            parents = random.sample(elites, k=2) if len(elites) >= 2 else random.choices(
                self._population, k=2
            )
            child_genes = self._crossover(parents[0].genes, parents[1].genes)
            child_genes = self._mutate(child_genes)
            new_pop.append(Genome(genes=child_genes, fitness=0.0))

        self._population = new_pop
        self._save_state_if_needed()

    def _crossover(self, g1: Dict[str, float], g2: Dict[str, float]) -> Dict[str, float]:
        child: Dict[str, float] = {}
        for aid in g1.keys() | g2.keys():
            if random.random() < 0.5:
                child[aid] = g1.get(aid, 1.0)
            else:
                child[aid] = g2.get(aid, 1.0)
        return child

    def _mutate(self, genes: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for aid, w in genes.items():
            if random.random() < self.mutation_rate:
                w += random.uniform(-self.mutation_scale, self.mutation_scale)
            w = max(self.min_weight, min(self.max_weight, w))
            out[aid] = w
        return out

    # ---- Persistence -----------------------------------------------------

    def _save_state_if_needed(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [g.to_dict() for g in self._population]
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_state(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        self._population = [Genome.from_dict(d) for d in arr]

    # ---- Integration shortcut -------------------------------------------

    def evolve(
        self,
        current_weights: Dict[str, float],
        stats: Dict[str, AgentStats],
    ) -> Dict[str, float]:
        """
        For online usage where no explicit fitness_map is provided, we can
        simply:
            - initialize population if empty
            - pick the best genome (or fallback to current)
            - (optionally) apply a tiny mutation to encourage exploration
        """
        agent_ids = list(current_weights.keys())
        if not self._population:
            self.initialize_population(agent_ids, current_weights)

        best = self.best_genome()
        if best is None:
            return current_weights

        genes = dict(best.genes)
        # tiny online mutation for exploration
        mutated = self._mutate(genes)
        logger.info("GeneticWeightEvolver selected genes: %s", mutated)
        return mutated


__all__ = [
    "BaseWeightEvolver",
    "HeuristicWeightEvolver",
    "RLWeightEvolver",
    "GeneticWeightEvolver",
    "Genome",
]
