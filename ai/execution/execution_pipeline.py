# ai/execution/execution_pipeline.py
"""
Execution pipeline for AITradeBot.

Phases integrated:
    • Phase 61 – MultiAgentBrain (decision fusion)
    • Phase 62 – ExecutionPipeline (brain → broker)
    • Phase 63 – Recorder (handled by ExecutionPipelineWithRecorder wrapper)
    • Phase 65 – Agent performance & evolution (consumes ExecutionResult)
    • Phase 67 – SafetyGuard (trade filters)

Key responsibilities:
    - Call MultiAgentBrain.decide(ctx) to get a FusedDecision
    - Decide whether to trade (BUY/SELL) or skip (HOLD / unsafe)
    - Apply SafetyGuard checks before routing orders
    - Delegate actual order submission to a BrokerExecutor
    - Return an ExecutionResult object for downstream components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging
import math

from ai.agents.base_agent import AgentContext
from ai.agents.votes import FusedDecision
from ai.agents.multi_agent_brain import MultiAgentBrain
from ai.execution.safety_guard import SafetyGuard, SafetyConfig
from ai.execution.order_validator import OrderValidator, ValidationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """
    Result of a pipeline pass.

    Attributes:
        decision:    FusedDecision produced by MultiAgentBrain
        order_sent:  True if an order was routed to broker
        order_meta:  Dict with broker response (status, ids, etc.)
    """

    decision: FusedDecision
    order_sent: bool
    order_meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Broker executors
# ---------------------------------------------------------------------------


class BaseBrokerExecutor:
    """
    Abstract adapter interface for brokers.

    Concrete implementations:
        - RouterBrokerExecutor: uses SmartOrderRouter
        - DummyBrokerExecutor: logs only (for DEMO/DRY_RUN)
    """

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        tag: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RouterBrokerExecutor(BaseBrokerExecutor):
    """
    Adapter that wraps SmartOrderRouter.

    Expected SmartOrderRouter API:
        router.route_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            tag=tag,
            meta=meta,
        ) -> dict
    """

    def __init__(self, router: Any) -> None:
        self.router = router
        self.logger = logging.getLogger("RouterBrokerExecutor")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        tag: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        try:
            self.logger.info(
                "🚀 Routing %s %s x %.4f via SmartOrderRouter (type=%s, tag=%s)",
                side,
                symbol,
                qty,
                order_type,
                tag,
            )
            resp = self.router.route_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price,
                tag=tag,
                meta=meta,
            )
            if not isinstance(resp, dict):
                resp = {"status": "UNKNOWN", "raw": resp}
            return resp
        except Exception as exc:
            self.logger.error("💥 RouterBrokerExecutor error: %s", exc, exc_info=True)
            return {
                "status": "ERROR",
                "error": str(exc),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "limit_price": limit_price,
                "tag": tag,
            }


class DummyBrokerExecutor(BaseBrokerExecutor):
    """
    Dummy broker for DEMO/DRY_RUN.

    Does not touch any real API. Only logs orders and returns a synthetic OK.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("DummyBrokerExecutor")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        tag: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = meta or {}
        self.logger.info(
            "🧪 DummyBroker: %s %.4f %s @ type=%s limit=%s tag=%s (meta keys=%s)",
            side,
            qty,
            symbol,
            order_type,
            limit_price,
            tag,
            list(meta.keys()),
        )
        return {
            "status": "DUMMY_OK",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "tag": tag,
        }


# ---------------------------------------------------------------------------
# Execution pipeline
# ---------------------------------------------------------------------------


class ExecutionPipeline:
    """
    Orchestrates:
        • MultiAgentBrain decision
        • SafetyGuard filters
        • Broker executor
        • ExecutionResult to feed recorder & evolution manager

    Public API:
        result = pipeline.decide_and_execute(ctx)
    """

    def __init__(
        self,
        *,
        brain: MultiAgentBrain,
        broker: BaseBrokerExecutor,
        min_confidence: float = 0.4,
        min_qty: float = 1.0,
        tag: str = "phase26",
        safety_cfg: Optional[Dict[str, Any]] = None,
        order_validator: Optional[OrderValidator] = None,
        account_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            brain:          MultiAgentBrain instance
            broker:         Broker executor (RouterBrokerExecutor or DummyBrokerExecutor)
            min_confidence: Minimum fused_conf required before even considering a trade
            min_qty:        Minimum absolute quantity to trade
            tag:            Tag used in broker meta (e.g. 'phase26_live')
            safety_cfg:     Optional dict to configure SafetyGuard (Phase 67)
        """
        self.brain = brain
        self.broker = broker
        self.min_confidence = float(min_confidence)
        self.min_qty = float(min_qty)
        self.tag = str(tag)
        self.logger = logging.getLogger("ExecutionPipeline")
        self.account_provider = account_provider

        # --- Phase 67 SafetyGuard configuration ---
        safety_cfg = safety_cfg or {}
        sc = SafetyConfig(
            min_confidence=float(safety_cfg.get("min_confidence", 0.55)),
            max_conflict=float(safety_cfg.get("max_conflict", 0.60)),
            max_position_qty=float(safety_cfg.get("max_position_qty", 500.0)),
            max_trade_qty=float(safety_cfg.get("max_trade_qty", 300.0)),
            max_volatility=float(safety_cfg.get("max_volatility", 0.35)),
            max_drawdown=float(safety_cfg.get("max_drawdown", 0.25)),
            flip_flop_window_sec=int(safety_cfg.get("flip_flop_sec", 45)),
            max_trades_per_min=int(safety_cfg.get("max_trades_per_min", 6)),
        )
        self.safety_guard = SafetyGuard(sc)
        
        # Order validator
        validator_cfg = ValidationConfig(
            min_qty=min_qty,
            max_qty=float(safety_cfg.get("max_trade_qty", 300.0)),
        )
        self.order_validator = order_validator or OrderValidator(config=validator_cfg)

        self.logger.info(
            "ExecutionPipeline initialized (tag=%s, min_conf=%.3f, min_qty=%.2f, safety=%s)",
            self.tag,
            self.min_confidence,
            self.min_qty,
            sc,
        )

    # ------------------------------------------------------------------ #
    # Public method
    # ------------------------------------------------------------------ #

    def decide_and_execute(self, ctx: AgentContext) -> ExecutionResult:
        """
        Main entrypoint for realtime loop.

        1. Ask brain for decision.
        2. Decide whether to execute or skip.
        3. If executing, apply SafetyGuard before routing.
        4. Return ExecutionResult for recorder & evolution.
        """
        decision = self.brain.decide(ctx)
        symbol = ctx.symbol
        final_action = decision.final_action
        final_size = decision.final_size
        fused_conf = decision.fused_conf
        conflict_score = decision.conflict_score

        # Default meta for non-executed cases
        order_meta: Dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "",
            "symbol": symbol,
            "side": final_action,
            "qty": final_size,
            "conf": fused_conf,
            "conflict": conflict_score,
        }

        # 1) No trade if HOLD
        if final_action not in ("BUY", "SELL"):
            order_meta["reason"] = "No trade: final_action is HOLD"
            self.logger.info(
                "⏸️ No trade for %s – final_action=%s (conf=%.3f, conflict=%.3f)",
                symbol,
                final_action,
                fused_conf,
                conflict_score,
            )
            return ExecutionResult(decision=decision, order_sent=False, order_meta=order_meta)

        # 2) No trade if size missing or tiny
        if final_size is None or abs(final_size) < self.min_qty:
            order_meta["reason"] = f"No trade: final_size={final_size} below min_qty={self.min_qty}"
            self.logger.info(
                "⏸️ No trade for %s – size too small %.4f (min=%.2f)",
                symbol,
                0.0 if final_size is None else final_size,
                self.min_qty,
            )
            return ExecutionResult(decision=decision, order_sent=False, order_meta=order_meta)

        # 3) Optional pre-filter by pipeline min_confidence
        if fused_conf < self.min_confidence:
            order_meta["reason"] = f"No trade: fused_conf={fused_conf:.3f} < min_confidence={self.min_confidence:.3f}"
            self.logger.info(
                "⏸️ No trade for %s – low fused_conf %.3f < %.3f",
                symbol,
                fused_conf,
                self.min_confidence,
            )
            return ExecutionResult(decision=decision, order_sent=False, order_meta=order_meta)

        # 4) Apply SafetyGuard (Phase 67)
        # Get context features for safety checks
        price = float(ctx.price or 0.0)
        current_qty = float(ctx.position.get("qty", 0.0))
        equity = float(ctx.portfolio.get("equity", 0.0))
        max_drawdown = float(ctx.portfolio.get("max_drawdown", 0.0))
        volatility = float(ctx.extra.get("volatility", 0.2))

        ok, reason = self.safety_guard.allow_trade(
            symbol=symbol,
            side=final_action,
            qty=final_size,
            price=price,
            confidence=fused_conf,
            conflict=conflict_score,
            volatility=volatility,
            current_qty=current_qty,
            equity=equity,
            max_drawdown=max_drawdown,
        )

        if not ok:
            self.logger.warning("🛑 SafetyGuard: Trade BLOCKED for %s: %s", symbol, reason)
            order_meta["status"] = "BLOCKED"
            order_meta["reason"] = reason
            return ExecutionResult(decision=decision, order_sent=False, order_meta=order_meta)

        # 4) Order validation (pre-trade checks)
        order_dict = {
            "symbol": symbol,
            "side": final_action,
            "qty": abs(final_size),
            "order_type": "MARKET",
            "price": price if price > 0 else None,
        }
        
        account = None
        if self.account_provider:
            try:
                account = self.account_provider()
            except Exception as e:
                self.logger.warning("Failed to get account for validation: %s", e)
        
        is_valid, validation_error = self.order_validator.validate(
            order_dict,
            account=account,
            last_price=price if price > 0 else None,
        )
        
        if not is_valid:
            order_meta["reason"] = f"Order validation failed: {validation_error}"
            order_meta["status"] = "VALIDATION_FAILED"
            self.logger.warning("⛔ Order validation failed for %s %s: %s", final_action, symbol, validation_error)
            return ExecutionResult(decision=decision, order_sent=False, order_meta=order_meta)

        # 5) Route order
        # Determine broker and build meta
        broker_hint = decision.final_broker or "alpaca"
        qty = abs(final_size)

        self.logger.info(
            "🚀 Sending %s %s x %.2f via %s (conf=%.3f, conflict=%.3f, broker_hint=%s)",
            final_action,
            symbol,
            qty,
            type(self.broker).__name__,
            fused_conf,
            conflict_score,
            broker_hint,
        )

        broker_meta = {
            "symbol": symbol,
            "final_action": final_action,
            "final_size": final_size,
            "final_broker": broker_hint,
            "fused_conf": fused_conf,
            "conflict_score": conflict_score,
            "decided_at": decision.decided_at.isoformat(),
            "votes": [v.to_dict() for v in decision.votes],
            "meta": decision.meta,
            "execution": {
                "price": price,
                "volatility": volatility,
                "equity": equity,
                "max_drawdown": max_drawdown,
                "current_qty": current_qty,
            },
        }

        try:
            resp = self.broker.submit_order(
                symbol=symbol,
                side=final_action,
                qty=qty,
                order_type="MARKET",
                limit_price=None,
                tag=self.tag,
                meta=broker_meta,
            )
        except Exception as exc:
            self.logger.error("💥 BrokerExecutor exception for %s: %s", symbol, exc, exc_info=True)
            resp = {
                "status": "ERROR",
                "error": str(exc),
                "symbol": symbol,
                "side": final_action,
                "qty": qty,
            }

        status = str(resp.get("status", "UNKNOWN"))
        if status.startswith("ERROR"):
            self.logger.error("💥 Order failed for %s: %s", symbol, resp)
        else:
            self.logger.info("✅ Order submitted for %s: %s", symbol, resp)

        return ExecutionResult(
            decision=decision,
            order_sent=not status.startswith("ERROR") and status not in ("BLOCKED", "SKIPPED"),
            order_meta=resp,
        )

    # Backwards-compatible alias (if older code uses this name)
    def decide_execute(self, ctx: AgentContext) -> ExecutionResult:
        return self.decide_and_execute(ctx)


__all__ = [
    "ExecutionResult",
    "BaseBrokerExecutor",
    "RouterBrokerExecutor",
    "DummyBrokerExecutor",
    "ExecutionPipeline",
]
