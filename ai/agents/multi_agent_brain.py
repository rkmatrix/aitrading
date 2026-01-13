# ai/agents/multi_agent_brain.py
"""
Phase 61+65 – Multi-Agent Trade Brain (MATB)

Coordinates multiple decision agents and fuses their votes into a single
final decision with:
    - weighted voting
    - conflict scoring
    - risk overrides
    - broker + size hints
    - human-readable justification

Phase 65 adds:
    - get_agent_weights()
    - set_agent_weights(mapping)
So evolution engines can dynamically adjust agent weights.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .base_agent import AgentContext, BaseAgent
from .votes import ACTIONS, AgentVote, FusedDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple built-in reference agents
# ---------------------------------------------------------------------------


class RandomRLAgent(BaseAgent):
    """
    Very simple RL-style agent placeholder.

    Tuned for DEMO:
        • More sensitive to drift
        • More likely to BUY/SELL when there is a trend
    """

    def __init__(self, agent_id: str = "rl_main", **kwargs: Any) -> None:
        super().__init__(agent_id, agent_type="rl", name="RandomRLAgent", **kwargs)

    def get_vote(self, ctx: AgentContext) -> AgentVote:
        drift = float(ctx.extra.get("price_drift", 0.0))

        # Much more sensitive thresholds now
        if drift > 0.002:      # ~0.2% up move
            action = "BUY"
        elif drift < -0.002:   # ~0.2% down move
            action = "SELL"
        else:
            # If drift is tiny, still occasionally take a side
            action = random.choices(
                ["BUY", "SELL", "HOLD"],
                weights=[0.3, 0.3, 0.4],
            )[0]

        # Confidence grows with |drift|
        confidence = min(1.0, 0.4 + abs(drift) * 80.0)
        reason = f"RL-style heuristic on drift={drift:.4f}"

        return AgentVote(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=ctx.symbol,
            action=action,
            confidence=confidence,
            weight=self.weight,
            reason=reason,
            meta={"drift": drift},
        )



class TrendSignalAgent(BaseAgent):
    """
    Uses simple moving-average-like hints from ctx.extra to decide direction.

    Tuned for DEMO:
        • More sensitive to small MA differences
        • Generates BUY/SELL more frequently when there is a trend
    """

    def __init__(self, agent_id: str = "signal_trend", **kwargs: Any) -> None:
        super().__init__(agent_id, agent_type="signal", name="TrendSignalAgent", **kwargs)

    def get_vote(self, ctx: AgentContext) -> AgentVote:
        short_ma = float(ctx.extra.get("short_ma", ctx.price or 0.0))
        long_ma = float(ctx.extra.get("long_ma", ctx.price or 0.0))

        if long_ma <= 0:
            long_ma = max(short_ma, 1e-6)

        ratio = short_ma / long_ma

        # Narrower band around 1.0 → more BUY/SELL signals
        if ratio > 1.001:        # short > long by 0.1%+
            action = "BUY"
        elif ratio < 0.999:      # short < long by 0.1%+
            action = "SELL"
        else:
            action = "HOLD"

        spread = abs(short_ma - long_ma)
        base = max(short_ma, long_ma, 1e-9)
        confidence = min(1.0, 0.3 + (spread / base) * 30.0)

        reason = f"ratio={ratio:.4f}, short_ma={short_ma:.2f}, long_ma={long_ma:.2f}"

        return AgentVote(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=ctx.symbol,
            action=action,
            confidence=confidence,
            weight=self.weight,
            reason=reason,
            meta={"short_ma": short_ma, "long_ma": long_ma, "ratio": ratio},
        )



class SimpleRiskAgent(BaseAgent):
    """
    Conservative risk agent that can override to HOLD when conditions are risky.

    Uses:
        - ctx.portfolio["max_drawdown"] (optional)
        - ctx.extra["volatility"] (optional)
        - ctx.position["qty"] (optional)
    """

    def __init__(self, agent_id: str = "risk_guardian", **kwargs: Any) -> None:
        super().__init__(agent_id, agent_type="risk", name="SimpleRiskAgent", **kwargs)

    def get_vote(self, ctx: AgentContext) -> AgentVote:
        max_dd = float(ctx.portfolio.get("max_drawdown", 0.0))
        vol = float(ctx.extra.get("volatility", 0.0))
        qty = float(ctx.position.get("qty", 0.0))

        risk_score = 0.0
        risk_score += max(0.0, max_dd) * 2.0
        risk_score += vol * 3.0
        risk_score += abs(qty) * 0.001

        if risk_score > 1.0:
            action = "HOLD"
            confidence = min(1.0, risk_score / 2.0)
            reason = f"Risk score {risk_score:.3f} above threshold → HOLD"
        else:
            action = "HOLD"
            confidence = 0.3
            reason = f"Risk score {risk_score:.3f} below threshold → neutral HOLD"

        return AgentVote(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=ctx.symbol,
            action=action,
            confidence=confidence,
            weight=self.weight,
            reason=reason,
            meta={"max_drawdown": max_dd, "volatility": vol, "qty": qty, "risk_score": risk_score},
        )


class SimpleAllocatorAgent(BaseAgent):
    """
    Recommends a position size based on equity and volatility.

    Expects:
        - ctx.portfolio["equity"]
        - ctx.extra["volatility"]
    """

    def __init__(self, agent_id: str = "allocator_simple", **kwargs: Any) -> None:
        super().__init__(agent_id, agent_type="allocator", name="SimpleAllocatorAgent", **kwargs)

    def get_vote(self, ctx: AgentContext) -> AgentVote:
        equity = float(ctx.portfolio.get("equity", 0.0))
        price = float(ctx.price or 0.0)
        vol = float(ctx.extra.get("volatility", 0.2)) or 0.2

        if equity <= 0 or price <= 0:
            size = 0.0
            confidence = 0.2
            reason = "Missing equity or price; size=0"
        else:
            target_risk = 0.01
            dollar_risk = equity * target_risk
            adj = max(vol, 0.05)
            dollar_alloc = dollar_risk / adj
            size = dollar_alloc / price
            size = max(0.0, size)
            confidence = 0.7
            reason = f"Equity={equity:.2f}, price={price:.2f}, vol={vol:.3f} → size≈{size:.2f}"

        return AgentVote(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=ctx.symbol,
            action="HOLD",
            confidence=confidence,
            weight=self.weight,
            size=size,
            reason=reason,
            meta={"equity": equity, "price": price, "volatility": vol},
        )


class SimpleRouterAgent(BaseAgent):
    """
    Gives a broker preference based on a pre-ranked list and optional scores.

    Expects:
        - ctx.extra["router_scores"] → {broker: score}
    """

    def __init__(
        self,
        agent_id: str = "router_pref",
        *,
        brokers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id, agent_type="router", name="SimpleRouterAgent", **kwargs)
        self.brokers = brokers or ["alpaca", "backup"]

    def get_vote(self, ctx: AgentContext) -> AgentVote:
        scores = ctx.extra.get("router_scores", {}) or {}
        best_broker = None
        best_score = -math.inf

        for b in self.brokers:
            s = float(scores.get(b, 0.0))
            if s > best_score:
                best_score = s
                best_broker = b

        if best_broker is None and self.brokers:
            best_broker = self.brokers[0]
            best_score = 0.0

        confidence = min(1.0, max(0.1, (best_score + 1.0) / 2.0))

        reason = f"broker_scores={scores}, selected={best_broker}, score={best_score:.3f}"

        return AgentVote(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=ctx.symbol,
            action="HOLD",
            confidence=confidence,
            weight=self.weight,
            broker_hint=best_broker,
            reason=reason,
            meta={"scores": scores},
        )


# ---------------------------------------------------------------------------
# MultiAgentBrain core
# ---------------------------------------------------------------------------


class MultiAgentBrain:
    """
    Central ensemble for Phase 61, extended in Phase 65 to allow dynamic
    agent weight updates.

    Usage:
        brain = MultiAgentBrain.from_simple_config(cfg)
        weights = brain.get_agent_weights()
        brain.set_agent_weights(new_weights)
        decision = brain.decide(ctx)
    """

    def __init__(
        self,
        *,
        conflict_threshold: float = 0.6,
        risk_hold_override: bool = False,
    ) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self.conflict_threshold = float(conflict_threshold)
        self.risk_hold_override = bool(risk_hold_override)

    # ---- Agent management -------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        if agent.agent_id in self.agents:
            logger.warning("Overwriting existing agent with id=%s", agent.agent_id)
        self.agents[agent.agent_id] = agent
        logger.info("🧠 Registered agent: %s (%s)", agent.agent_id, agent.agent_type)

    def register_many(self, agents: Iterable[BaseAgent]) -> None:
        for a in agents:
            self.register_agent(a)

    # ---- Phase 65: dynamic weight helpers --------------------------------

    def get_agent_weights(self) -> Dict[str, float]:
        """
        Returns a mapping of agent_id -> current weight.
        """
        return {aid: float(a.weight) for aid, a in self.agents.items()}

    def set_agent_weights(self, weights: Dict[str, float]) -> None:
        """
        Sets agent weights from a mapping: agent_id -> weight.
        Agents not present in mapping keep their existing weights.
        """
        for aid, w in weights.items():
            if aid in self.agents:
                self.agents[aid].weight = float(w)
        logger.info("MultiAgentBrain updated agent weights: %s", self.get_agent_weights())

    # ---- Decision pipeline ------------------------------------------------

    def collect_votes(self, ctx: AgentContext) -> List[AgentVote]:
        votes: List[AgentVote] = []
        for agent_id, agent in self.agents.items():
            vote = agent.safe_get_vote(ctx)
            if vote is not None:
                votes.append(vote)
        return votes

    def _aggregate_actions(self, votes: List[AgentVote]) -> Dict[str, float]:
        scores = {a: 0.0 for a in ACTIONS}
        for v in votes:
            if v.action not in scores:
                continue
            scores[v.action] += max(0.0, float(v.confidence)) * max(0.0, float(v.weight))
        return scores

    def _compute_conflict(self, scores: Dict[str, float]) -> float:
        total = sum(scores.values())
        if total <= 1e-9:
            return 0.0
        vals = [s / total for s in scores.values()]
        max_val = max(vals)
        conflict = 1.0 - max_val
        return max(0.0, min(1.0, conflict))

    def _choose_final_action(self, scores: Dict[str, float]) -> str:
        best_action = "HOLD"
        best_score = -math.inf
        for a in ACTIONS:
            s = scores.get(a, 0.0)
            if s > best_score:
                best_score = s
                best_action = a
        return best_action

    def _aggregate_size(self, votes: List[AgentVote]) -> Optional[float]:
        num = 0.0
        den = 0.0
        for v in votes:
            if v.size is None:
                continue
            if v.agent_type not in ("allocator", "rl"):
                continue
            w = max(0.0, float(v.weight)) * max(0.0, float(v.confidence))
            num += w * float(v.size)
            den += w
        if den <= 1e-9:
            return None
        return num / den

    def _aggregate_broker(self, votes: List[AgentVote]) -> Optional[str]:
        score_by_broker: Dict[str, float] = {}
        for v in votes:
            if not v.broker_hint:
                continue
            w = max(0.0, float(v.weight)) * max(0.0, float(v.confidence))
            score_by_broker[v.broker_hint] = score_by_broker.get(v.broker_hint, 0.0) + w
        if not score_by_broker:
            return None
        return max(score_by_broker.items(), key=lambda kv: kv[1])[0]

    def _apply_risk_overrides(
        self,
        votes: List[AgentVote],
        final_action: str,
        conflict: float,
    ) -> str:
        if self.risk_hold_override:
            for v in votes:
                if v.agent_type == "risk" and v.action == "HOLD" and v.confidence >= 0.7:
                    logger.info("🛡️ Risk override: forcing HOLD due to %s", v.agent_id)
                    return "HOLD"

        if conflict >= self.conflict_threshold:
            logger.info("⚖️ High conflict (%.3f ≥ %.3f) → forcing HOLD", conflict, self.conflict_threshold)
            return "HOLD"

        return final_action

    def decide(self, ctx: AgentContext) -> FusedDecision:
        votes = self.collect_votes(ctx)
        if not votes:
            logger.warning("No agent votes available; defaulting to HOLD for %s", ctx.symbol)
            scores = {a: 0.0 for a in ACTIONS}
            scores["HOLD"] = 1.0
        else:
            scores = self._aggregate_actions(votes)

        conflict = self._compute_conflict(scores)
        raw_action = self._choose_final_action(scores)
        final_action = self._apply_risk_overrides(votes, raw_action, conflict)

        total_score = sum(scores.values())
        final_score = scores.get(final_action, 0.0)
        fused_conf = 0.0 if total_score <= 1e-9 else max(0.0, min(1.0, final_score / total_score))

        size = self._aggregate_size(votes)
        broker = self._aggregate_broker(votes)

        meta = {
            "scores": scores,
            "raw_action": raw_action,
            "risk_hold_override": self.risk_hold_override,
            "conflict_threshold": self.conflict_threshold,
        }

        decision = FusedDecision(
            symbol=ctx.symbol,
            final_action=final_action,
            final_size=size,
            final_broker=broker,
            fused_conf=fused_conf,
            conflict_score=conflict,
            votes=votes,
            decided_at=datetime.utcnow(),
            meta=meta,
        )

        self._log_decision(decision)
        return decision

    def _log_decision(self, decision: FusedDecision) -> None:
        logger.info(
            "🧠 Phase61 decision for %s: action=%s size=%s broker=%s conf=%.3f conflict=%.3f",
            decision.symbol,
            decision.final_action,
            f"{decision.final_size:.2f}" if decision.final_size is not None else "None",
            decision.final_broker or "None",
            decision.fused_conf,
            decision.conflict_score,
        )

        if logger.isEnabledFor(logging.DEBUG):
            for v in decision.votes:
                logger.debug(
                    "  • %-14s [%-8s] act=%-4s conf=%.3f w=%.2f size=%s broker=%s | %s",
                    v.agent_id,
                    v.agent_type,
                    v.action,
                    v.confidence,
                    v.weight,
                    f"{v.size:.2f}" if v.size is not None else "None",
                    v.broker_hint or "None",
                    v.reason,
                )

    # ---- Factory ----------------------------------------------------------

    @classmethod
    def from_simple_config(cls, cfg: Dict[str, Any]) -> "MultiAgentBrain":
        conflict_threshold = float(cfg.get("conflict_threshold", 0.6))
        risk_hold_override = bool(cfg.get("risk_hold_override", True))

        brain = cls(
            conflict_threshold=conflict_threshold,
            risk_hold_override=risk_hold_override,
        )

        a_cfg = cfg.get("agents", {}) or {}
        rl_weight = float(a_cfg.get("rl", {}).get("weight", 1.0))
        sig_weight = float(a_cfg.get("signal", {}).get("weight", 1.0))
        risk_weight = float(a_cfg.get("risk", {}).get("weight", 1.5))
        alloc_weight = float(a_cfg.get("allocator", {}).get("weight", 1.0))
        router_weight = float(a_cfg.get("router", {}).get("weight", 0.8))

        router_brokers = a_cfg.get("router", {}).get("brokers", ["alpaca", "backup"])

        brain.register_many(
            [
                RandomRLAgent(weight=rl_weight),
                TrendSignalAgent(weight=sig_weight),
                SimpleRiskAgent(weight=risk_weight),
                SimpleAllocatorAgent(weight=alloc_weight),
                SimpleRouterAgent(weight=router_weight, brokers=router_brokers),
            ]
        )

        return brain


__all__ = [
    "MultiAgentBrain",
    "RandomRLAgent",
    "TrendSignalAgent",
    "SimpleRiskAgent",
    "SimpleAllocatorAgent",
    "SimpleRouterAgent",
]
