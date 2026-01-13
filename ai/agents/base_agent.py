# ai/agents/base_agent.py
"""
Phase 61 – Base Agent Abstractions

Provides:
    - AgentContext: shared context passed to all agents
    - BaseAgent:    abstract base class for all decision agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import abc
import logging

from .votes import AgentVote

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context passed into each agent when requesting a vote.

    Fields:
        symbol:     Target symbol to trade.
        ts:         Current timestamp (UTC).
        price:      Last known price (optional).
        position:   Dict with current symbol-level position info (qty, avg_cost, etc.).
        portfolio:  Dict with account/portfolio stats (equity, exposure, etc.).
        extra:      Free-form context from upstream modules (signals, features, etc.).
    """

    symbol: str
    ts: datetime
    price: Optional[float] = None
    position: Dict[str, Any] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all decision agents used by MultiAgentBrain.

    Subclasses must implement:
        - get_vote(self, ctx: AgentContext) -> AgentVote
    """

    def __init__(
        self,
        agent_id: str,
        *,
        agent_type: str,
        name: Optional[str] = None,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type  # "rl", "signal", "risk", "allocator", "router", etc.
        self.name = name or agent_id
        self.weight = float(weight)
        self.enabled = bool(enabled)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{self.__class__.__name__} id={self.agent_id} type={self.agent_type} weight={self.weight}>"

    @abc.abstractmethod
    def get_vote(self, ctx: AgentContext) -> AgentVote:
        """
        Produce an AgentVote for the given context.

        Must set:
            - symbol
            - action
            - confidence
        and optionally:
            - size
            - broker_hint
            - reason
            - meta
        """
        raise NotImplementedError

    def safe_get_vote(self, ctx: AgentContext) -> Optional[AgentVote]:
        """
        Wrapper to protect the ensemble from one agent crashing.

        If an exception occurs, it is logged and None is returned.
        """
        if not self.enabled:
            logger.debug("Agent %s disabled; skipping vote.", self.agent_id)
            return None

        try:
            vote = self.get_vote(ctx)
            # Ensure basic invariants
            vote.agent_id = self.agent_id
            vote.agent_type = self.agent_type
            vote.weight = self.weight
            return vote
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("💥 Agent %s failed to produce a vote: %s", self.agent_id, exc, exc_info=True)
            return None


__all__ = ["AgentContext", "BaseAgent"]
