# ai/evolution/agent_performance_tracker.py
"""
Phase 65 – Agent Performance Tracker

Tracks per-agent performance statistics over time. This is the data source
for the various weight evolution strategies (A: heuristic, B: RL-style, C: genetic).

This module is intentionally generic:
    - You feed in agent-level rewards via `update(agent_id, reward)`.
    - It keeps rolling stats (count, avg, variance, last reward, etc.).
    - You can persist and reload from JSON for long-running processes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    agent_id: str
    count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    last_reward: float = 0.0
    last_updated: Optional[str] = None  # ISO timestamp

    @property
    def mean_reward(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.reward_sum / self.count

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean_reward
        return max(0.0, self.reward_sq_sum / self.count - mean * mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentStats":
        return cls(**d)


class AgentPerformanceTracker:
    """
    Keeps track of AgentStats for each agent_id.

    Intended usage:

        tracker = AgentPerformanceTracker()
        tracker.update("rl_main", reward=+0.5)
        tracker.update("signal_trend", reward=-0.2)
        stats = tracker.get_all_stats()
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentStats] = {}

    # ------------------------------------------------------------------ #
    # Core updates
    # ------------------------------------------------------------------ #

    def update(self, agent_id: str, reward: float) -> None:
        """
        Update the performance stats for a given agent with a new reward.

        `reward` can be:
            - signed PnL contribution
            - correctness score (+1 correct / -1 incorrect)
            - any scalar signal you choose
        """
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentStats(agent_id=agent_id)

        s = self._agents[agent_id]
        s.count += 1
        s.reward_sum += float(reward)
        s.reward_sq_sum += float(reward) ** 2
        s.last_reward = float(reward)
        s.last_updated = datetime.utcnow().isoformat()

    def bulk_update(self, rewards: Dict[str, float]) -> None:
        """
        Convenient helper to update multiple agents at once.
        """
        for aid, r in rewards.items():
            self.update(aid, r)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_stats(self, agent_id: str) -> AgentStats:
        return self._agents.get(agent_id, AgentStats(agent_id=agent_id))

    def get_all_stats(self) -> Dict[str, AgentStats]:
        return dict(self._agents)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {aid: s.to_dict() for aid, s in self._agents.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentPerformanceTracker":
        tracker = cls()
        for aid, sd in d.items():
            tracker._agents[aid] = AgentStats.from_dict(sd)
        return tracker

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "AgentPerformanceTracker":
        p = Path(path)
        if not p.exists():
            logger.warning("No performance file at %s; starting fresh tracker.", p)
            return cls()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


__all__ = ["AgentPerformanceTracker", "AgentStats"]
