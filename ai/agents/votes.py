# ai/agents/votes.py
"""
Phase 61 – Multi-Agent Voting Structures

Defines:
    - AgentVote: single agent's decision (action, confidence, reason, etc.)
    - FusedDecision: final ensemble outcome from MultiAgentBrain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# Canonical action space for all agents
ACTIONS = ("BUY", "SELL", "HOLD")


@dataclass
class AgentVote:
    """
    Represents a single agent's vote/decision.

    Fields:
        agent_id:     Unique identifier (e.g. "rl_main", "risk_guardian")
        agent_type:   Category ("rl", "signal", "risk", "allocator", "router", ...)
        symbol:       Target symbol
        action:       "BUY" | "SELL" | "HOLD"
        confidence:   0.0–1.0 (agent's internal confidence)
        weight:       Relative influence in the ensemble (default 1.0)
        size:         Optional suggested size (shares or units)
        broker_hint:  Optional preferred broker or route
        reason:       Human-readable explanation
        meta:         Any extra agent-specific data
        created_at:   Timestamp when vote was created
    """

    agent_id: str
    agent_type: str
    symbol: str
    action: str
    confidence: float = 0.5
    weight: float = 1.0
    size: Optional[float] = None
    broker_hint: Optional[str] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "weight": self.weight,
            "size": self.size,
            "broker_hint": self.broker_hint,
            "reason": self.reason,
            "meta": self.meta,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FusedDecision:
    """
    Final ensemble decision from MultiAgentBrain.

    Fields:
        symbol:          Target symbol
        final_action:    "BUY" | "SELL" | "HOLD"
        final_size:      Final chosen size (if any)
        final_broker:    Broker/route to use (if any)
        fused_conf:      0.0–1.0 aggregate confidence
        conflict_score:  0.0–1.0 how much agents disagree (higher = more conflict)
        votes:           List of AgentVote objects
        decided_at:      Timestamp
        meta:            Optional additional info
    """

    symbol: str
    final_action: str
    final_size: Optional[float]
    final_broker: Optional[str]
    fused_conf: float
    conflict_score: float
    votes: List[AgentVote]
    decided_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "final_action": self.final_action,
            "final_size": self.final_size,
            "final_broker": self.final_broker,
            "fused_conf": self.fused_conf,
            "conflict_score": self.conflict_score,
            "decided_at": self.decided_at.isoformat(),
            "votes": [v.to_dict() for v in self.votes],
            "meta": self.meta,
        }


__all__ = ["ACTIONS", "AgentVote", "FusedDecision"]
