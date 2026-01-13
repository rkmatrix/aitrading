"""
ai/execution/smart_order_router.py
----------------------------------

SmartOrderRouter v4 (Phase 92.x)

Pipeline
--------
order → RiskEnvelope → (Prediction Gate) → MultiExchangeRouter → Broker

Features
--------
- RiskEnvelopeController (Phase 69C / 92.x)
- MultiExchangeRouter (Phase 69D)
- SlippagePredictor (Phase 73–74)
- OrderLedger / TradeJournal hooks (Phase 71+)
- Execution metrics (slippage, spread, latency)
- GuardrailRuntimeHandler (Phase 122.3 – runtime safety)
- ExecutionQualityMemory hook (Phase 123 – EQM)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from tools.telegram_alerts import notify
except Exception:  # optional
    def notify(
        msg: str,
        *,
        kind: str = "system",
        meta: Dict[str, Any] | None = None,
    ) -> None:
        logging.getLogger("SmartOrderRouter").info("TELEGRAM (stub): %s", msg)

from ai.risk.risk_envelope import RiskEnvelopeController
from ai.execution.multi_exchange_router import MultiExchangeRouter
from ai.execution.broker_alpaca_live import AlpacaLiveBroker
from ai.execution.slippage_predictor import SlippagePredictor
from ai.safety.auto_guardrails import GuardrailRuntimeHandler

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Light stubs for ledger / journal (if user modules missing)
# ----------------------------------------------------------------------


class OrderLedger:
    @staticmethod
    def order_submitted(*args, **kwargs) -> None:
        pass

    @staticmethod
    def order_error(*args, **kwargs) -> None:
        pass

    @staticmethod
    def order_complete(*args, **kwargs) -> None:
        pass

    @staticmethod
    def log(*args, **kwargs) -> None:
        pass


class TradeJournal:
    @staticmethod
    def append_entry(*args, **kwargs) -> None:
        pass


@dataclass
class RouteResult:
    status: str
    broker: Optional[str]
    broker_order_id: Optional[str]
    routed_order: Dict[str, Any]
    route_meta: Dict[str, Any]
    fill: Optional[Dict[str, Any]]
    execution_metrics: Optional[Dict[str, Any]]
    error: Optional[str] = None


class SmartOrderRouter:
    """
    SmartOrderRouter v4
    """

    def __init__(
        self,
        *,
        risk_cfg_path: str = "configs/phase69c_risk_envelope.yaml",
        multix_cfg_path: str = "configs/phase69d_multix.yaml",
        portfolio_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        primary_broker: Optional[Any] = None,
        guardrails: Optional[GuardrailRuntimeHandler] = None,
        slippage_model_path: str = "models/execution/slippage_model.joblib",
        enable_prediction_gate: bool = True,
        stability_mode: bool = True,
    ) -> None:

        self.log = logging.getLogger("SmartOrderRouter")
        self.portfolio_provider = portfolio_provider
        self.primary_broker = primary_broker or AlpacaLiveBroker.from_env()
        self.guardrails = guardrails
        # Phase 123 – optional stability / EQM integration
        self.stability_mode: bool = bool(stability_mode)
        self.eqm = None  # to be injected by caller (e.g. Phase 26 loop)

        # ----------------- Risk Envelope Controller (Phase 69C) -----------------
        self.risk_ctrl: Optional[RiskEnvelopeController] = None
        try:
            self.risk_ctrl = RiskEnvelopeController.from_yaml(
                risk_cfg_path,
                portfolio_provider=portfolio_provider,
            )
            self.log.info("🛡️ RiskEnvelopeController initialized from %s", risk_cfg_path)
        except Exception as e:
            self.log.warning("⚠️ RiskEnvelopeController unavailable: %s", e)
            self.risk_ctrl = None

        # ----------------- Multi-Exchange Router (Phase 69D) --------------------
        self.multi_router: Optional[MultiExchangeRouter] = None
        try:
            self.multi_router = MultiExchangeRouter.from_yaml(
                multix_cfg_path,
                primary_broker=self.primary_broker,
            )
            self.log.info("🧭 MultiExchangeRouter initialized from %s", multix_cfg_path)
        except Exception as e:
            self.log.warning("⚠️ MultiExchangeRouter unavailable: %s", e)
            self.multi_router = None

        # ----------------- Slippage / Execution Predictor (Phase 73–74) --------
        self.slippage_predictor: Optional[SlippagePredictor] = None
        self.enable_prediction_gate = enable_prediction_gate
        if enable_prediction_gate:
            try:
                self.slippage_predictor = SlippagePredictor.from_model(slippage_model_path)
                self.log.info("📈 SlippagePredictor loaded from %s", slippage_model_path)
            except Exception as e:
                self.log.warning("⚠️ SlippagePredictor unavailable: %s", e)
                self.slippage_predictor = None

    # ======================================================================
    # Public API
    # ======================================================================
    def route_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        tag: Optional[str] = None,
        is_flattening: bool = False,
        journal_ctx: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        stop_price: float | None = None,
        client_tag: str | None = None,
        risk_ctx: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Route an order through:
            guardrails → risk envelope → prediction (optional) → router → broker.

        Returns:
            dict with keys:
                status: "OK" | "BLOCKED" | "ERROR"
                broker: broker name or None
                broker_order_id: str or None
                routed_order: dict
                route_meta: dict
                fill: dict or None
                execution_metrics: dict or None
                error: optional error string
        """
        start_ts = time.perf_counter()
        ts_utc = datetime.utcnow().isoformat()

        extra = extra or {}
        journal_ctx = journal_ctx or {}

        # ----------------- Guardrails (Phase 122.3) -----------------
        # Use basic quantity + notional metrics so guardrails can clamp/block
        if self.guardrails is not None:
            try:
                est_price = (
                    extra.get("price")
                    or extra.get("mid_price")
                    or extra.get("last_price")
                    or limit_price
                )
                notional = float(qty) * float(est_price) if est_price is not None else float(qty)
            except Exception:
                est_price = extra.get("price")
                notional = float(qty)

            decision = self.guardrails.check(
                event="route_order",
                metrics={
                    "qty": float(qty),
                    "notional": float(notional),
                },
                context={
                    "symbol": symbol,
                    "side": side,
                    "order_type": order_type,
                    "limit_price": limit_price,
                    "tag": tag,
                    "is_flattening": is_flattening,
                },
            )

            # Hard block
            if getattr(decision, "is_blocked", False):
                msg = "; ".join(decision.reasons) if getattr(decision, "reasons", None) else "Guardrails blocked order."
                self.log.warning(
                    "⛔ Guardrails blocked order %s %s x %.4f (notional=%.2f): %s",
                    side,
                    symbol,
                    float(qty),
                    float(notional),
                    msg,
                )
                try:
                    notify(
                        f"⛔ Guardrails BLOCKED order: {side} {qty} {symbol} (tag={tag})\n"
                        f"Reason: {msg}",
                        kind="internal",
                        meta={
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "notional": notional,
                            "guardrail_rules": getattr(decision, "rule_ids", []),
                        },
                    )
                except Exception:
                    pass

                # Journal entry for guardrail block
                minimal_order = {
                    "symbol": symbol,
                    "side": side,
                    "qty": float(qty),
                    "order_type": order_type,
                    "limit_price": limit_price,
                    "tag": tag,
                    "is_flattening": is_flattening,
                    "client_order_id": extra.get("client_order_id"),
                    "ts_created": ts_utc,
                }

                self._write_trade_journal_entry(
                    ts_utc=ts_utc,
                    order=minimal_order,
                    router_meta={
                        "blocked_by_guardrails": True,
                        "guardrail_rules": getattr(decision, "rule_ids", []),
                        "guardrail_reasons": getattr(decision, "reasons", []),
                    },
                    broker_name=None,
                    broker_order_id=None,
                    fill_price=None,
                    latency_ms=int((time.perf_counter() - start_ts) * 1000),
                    status="BLOCKED",
                    error=msg,
                    journal_ctx=journal_ctx,
                    extra=extra,
                )

                return asdict(
                    RouteResult(
                        status="BLOCKED",
                        broker=None,
                        broker_order_id=None,
                        routed_order=minimal_order,
                        route_meta={
                            "blocked": True,
                            "blocked_by_guardrails": True,
                            "guardrail_rules": getattr(decision, "rule_ids", []),
                            "guardrail_reasons": getattr(decision, "reasons", []),
                        },
                        fill=None,
                        execution_metrics=None,
                        error=msg,
                    )
                )

            # Soft clamp on qty
            if getattr(decision, "is_clamp", False) and getattr(decision, "metrics", None):
                new_qty = decision.metrics.get("qty", qty)
                if new_qty != qty:
                    self.log.info(
                        "Guardrails clamped qty for %s: %.4f → %.4f",
                        symbol,
                        float(qty),
                        float(new_qty),
                    )
                    qty = new_qty

        # Ensure we always carry a usable price for RiskEnvelope
        if "price" not in extra or extra.get("price") is None:
            inferred_price = extra.get("mid_price") or extra.get("last_price") or limit_price
            if inferred_price is not None:
                extra["price"] = inferred_price

        # Phase 123 – optional risk context from LiveCapitalGuardian
        if risk_ctx:
            self.log.info(
                "SmartOrderRouter: risk_ctx received for %s %s x %s → %s",
                side,
                symbol,
                qty,
                risk_ctx,
            )

        # Build internal order payload
        order = self._build_order_payload(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            tag=tag,
            is_flattening=is_flattening,
            client_order_id=extra.get("client_order_id"),
            ts=ts_utc,
            extra=extra,
        )

        OrderLedger.order_submitted(order, route_info={"stage": "pre_risk"})

        # ----------------- Risk Envelope -----------------
        risk_allowed, risk_order, risk_meta = self._apply_risk_envelope(order)
        OrderLedger.log(
            {
                "type": "risk_check",
                "order_id": order.get("client_order_id"),
                "allowed": risk_allowed,
                "risk_meta": risk_meta,
                "order_after_risk": risk_order,
            }
        )

        if not risk_allowed:
            msg = f"Order blocked by RiskEnvelope: {risk_meta.get('reason', 'unknown reason')}"
            self.log.warning("⛔ %s", msg)
            try:
                notify(
                    f"⛔ Order BLOCKED by risk: {side} {qty} {symbol} (tag={tag})\n"
                    f"Reason: {risk_meta.get('reason', 'n/a')}",
                    kind="internal",
                    meta={"order": risk_order, "risk_meta": risk_meta},
                )
            except Exception:
                pass

            OrderLedger.order_error(
                order.get("client_order_id"),
                msg,
                context={"risk_meta": risk_meta},
            )

            self._write_trade_journal_entry(
                ts_utc=ts_utc,
                order=risk_order,
                router_meta={"blocked_by_risk": True, "risk_meta": risk_meta},
                broker_name=None,
                broker_order_id=None,
                fill_price=None,
                latency_ms=int((time.perf_counter() - start_ts) * 1000),
                status="BLOCKED",
                error=msg,
                journal_ctx=journal_ctx,
                extra=extra,
            )

            result = RouteResult(
                status="BLOCKED",
                broker=None,
                broker_order_id=None,
                routed_order=risk_order,
                route_meta={"risk": risk_meta, "blocked": True},
                fill=None,
                execution_metrics=None,
                error=msg,
            )
            return asdict(result)

        routed_order = risk_order
        prediction_meta: Dict[str, Any] = {}

        # ----------------- Prediction Gate (if enabled) -----------------
        blocked_by_prediction = False
        pred_block_reason: Optional[str] = None
        if self.enable_prediction_gate and self.slippage_predictor is not None:
            routed_order, blocked_by_prediction, pred_block_reason = self._apply_prediction_gate(
                routed_order, extra, prediction_meta
            )

        if blocked_by_prediction:
            msg = f"Order blocked by prediction gate: {pred_block_reason or 'unknown'}"
            self.log.warning("⛔ %s", msg)
            OrderLedger.order_error(
                order.get("client_order_id"),
                msg,
                context={"prediction_meta": prediction_meta},
            )

            self._write_trade_journal_entry(
                ts_utc=ts_utc,
                order=routed_order,
                router_meta={
                    "blocked_by_prediction": True,
                    "prediction_meta": prediction_meta,
                    "risk_meta": risk_meta,
                },
                broker_name=None,
                broker_order_id=None,
                fill_price=None,
                latency_ms=int((time.perf_counter() - start_ts) * 1000),
                status="BLOCKED",
                error=msg,
                journal_ctx=journal_ctx,
                extra=extra,
            )

            result = RouteResult(
                status="BLOCKED",
                broker=None,
                broker_order_id=None,
                routed_order=routed_order,
                route_meta={
                    "risk": risk_meta,
                    "prediction": prediction_meta,
                    "blocked": True,
                    "blocked_by_prediction": True,
                    "block_reason": pred_block_reason,
                },
                fill=None,
                execution_metrics=None,
                error=msg,
            )
            return asdict(result)

        # ----------------- Routing Decision (Multi-Exchange) ------------------
        try:
            if self.multi_router:
                broker_name, broker_client, route_info = self.multi_router.choose_broker(routed_order)
            else:
                broker_name = "primary"
                broker_client = self.primary_broker
                route_info = {
                    "strategy": "primary_only",
                    "rank": 1,
                    "score": 1.0,
                    "all_scores": {"primary": 1.0},
                }

            route_meta = {
                "broker": broker_name,
                "router_info": route_info,
                "risk_meta": risk_meta,
                "prediction_meta": prediction_meta or None,
            }
        except Exception as e:
            msg = f"Routing decision failed: {e}"
            self.log.error("💥 %s", msg)
            OrderLedger.order_error(
                order.get("client_order_id"),
                msg,
                context={"exception": str(e)},
            )

            self._write_trade_journal_entry(
                ts_utc=ts_utc,
                order=routed_order,
                router_meta={"error": "router_fail", "risk_meta": risk_meta},
                broker_name=None,
                broker_order_id=None,
                fill_price=None,
                latency_ms=int((time.perf_counter() - start_ts) * 1000),
                status="ERROR",
                error=msg,
                journal_ctx=journal_ctx,
                extra=extra,
            )

            result = RouteResult(
                status="ERROR",
                broker=None,
                broker_order_id=None,
                routed_order=routed_order,
                route_meta={"risk": risk_meta, "router_error": True},
                fill=None,
                execution_metrics=None,
                error=msg,
            )
            return asdict(result)

        OrderLedger.log(
            {
                "type": "route_decision",
                "order_id": order.get("client_order_id"),
                "broker": broker_name,
                "router_info": route_info,
                "risk_meta": risk_meta,
                "prediction_meta": prediction_meta or None,
            }
        )

        # ----------------- Submit to broker -----------------
        submit_start = time.perf_counter()
        broker_resp = self._submit_to_broker(broker_client, routed_order)
        submit_latency_ms = int((time.perf_counter() - submit_start) * 1000)

        broker_order_id = self._extract_broker_order_id(broker_resp)
        route_meta["submit_latency_ms"] = submit_latency_ms
        route_meta["raw_response_type"] = type(broker_resp).__name__

        self.log.info(
            "🚀 Routing %s %s@%s via %s (id=%s, rank=%s, score=%.3f)",
            routed_order["side"],
            routed_order["qty"],
            routed_order["symbol"],
            broker_name,
            order.get("client_order_id"),
            route_info.get("rank"),
            route_info.get("score", 0.0),
        )

        # ----------------- Detect broker-level error (e.g. insufficient BP) ---
        broker_error: Optional[str] = None
        if isinstance(broker_resp, dict):
            broker_error = broker_resp.get("error")
            if broker_error is None and broker_resp.get("order_submitted") is False:
                broker_error = "order_submitted=False"

        if broker_error:
            self.log.error("💥 Broker %s error: %s", broker_name, broker_error)
            route_meta["broker_error"] = broker_error

        # ----------------- Fill & Execution Metrics (Phase 88.4) --------------
        fill = self._extract_fill(broker_resp)
        fill_price = fill.get("price") if fill else None
        spread = extra.get("spread")
        slippage = None
        if fill_price is not None and extra.get("mid_price") is not None:
            mid = float(extra["mid_price"])
            slippage = float(fill_price) - mid

        total_latency_ms = int((time.perf_counter() - start_ts) * 1000)

        exec_metrics = {
            "slippage": float(slippage) if slippage is not None else 0.0,
            "spread": float(spread) if spread is not None else 0.0,
            "submit_latency_ms": submit_latency_ms,
            "total_latency_ms": total_latency_ms,
        }

        # Phase 123 – Feed execution metrics into EQM (if available)
        try:
            if getattr(self, "eqm", None) is not None and routed_order is not None:
                sym = routed_order.get("symbol")
                if sym:
                    self.eqm.record(
                        symbol=sym,
                        metrics=dict(exec_metrics),
                        context={
                            "side": routed_order.get("side"),
                            "qty": routed_order.get("qty"),
                            "tag": routed_order.get("tag"),
                        },
                    )
        except Exception:
            self.log.exception("Phase 123 EQM: failed to record execution metrics.")

        # Ledger + journal, status depending on broker_error
        if broker_error:
            status = "ERROR"
            OrderLedger.order_error(
                order.get("client_order_id"),
                broker_error,
                context={"route_meta": route_meta, "exec_metrics": exec_metrics},
            )
        else:
            status = "OK"
            OrderLedger.order_complete(
                order.get("client_order_id"),
                summary={
                    "broker": broker_name,
                    "broker_order_id": broker_order_id,
                    "fill_price": fill_price,
                    "slippage": slippage,
                    "spread": spread,
                    "latency_ms": total_latency_ms,
                    "route_meta": route_meta,
                    "journal_ctx": journal_ctx,
                },
            )

        self._write_trade_journal_entry(
            ts_utc=ts_utc,
            order=routed_order,
            router_meta={**route_meta, "risk_meta": risk_meta, "prediction_meta": prediction_meta or None},
            broker_name=broker_name,
            broker_order_id=broker_order_id,
            fill_price=fill_price,
            latency_ms=total_latency_ms,
            status=status,
            error=broker_error,
            journal_ctx=journal_ctx,
            extra=extra,
        )

        result = RouteResult(
            status=status,
            broker=broker_name,
            broker_order_id=broker_order_id,
            routed_order=routed_order,
            route_meta=route_meta,
            fill=fill,
            execution_metrics=exec_metrics,
            error=broker_error,
        )
        return asdict(result)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _build_order_payload(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float],
        tag: Optional[str],
        is_flattening: bool,
        client_order_id: Optional[str],
        ts: str,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not client_order_id:
            client_order_id = f"{symbol}-{side}-{int(time.time() * 1000)}"

        order = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": float(qty),
            "order_type": order_type.upper(),
            "limit_price": limit_price,
            "tag": tag,
            "is_flattening": bool(is_flattening),
            "client_order_id": client_order_id,
            "ts_created": ts,
        }
        order.update({k: v for k, v in extra.items() if k not in order})
        return order

    def _submit_to_broker(self, broker_client: Any, order: Dict[str, Any]) -> Any:
        """
        Supports multiple broker adapter styles:

        - AlpacaLiveBroker.submit_order(order_dict)
        - Any client exposing .submit_order(order_dict)
        - Any client exposing .place_order(order_dict)

        Returns underlying response or a dict with "error".
        """
        if broker_client is None:
            self.log.error("❌ No broker_client provided to SmartOrderRouter.")
            return {"error": "no_broker_client", "order_submitted": False}

        try:
            # Preferred: our AlpacaLiveBroker wrapper
            if hasattr(broker_client, "submit_order"):
                return broker_client.submit_order(order)

            # Generic: any simulated or paper broker expecting order dict
            if hasattr(broker_client, "place_order"):
                return broker_client.place_order(order)

            # No supported method found
            self.log.error(
                "❌ Broker client %s has no supported submit method "
                "(expected submit_order / place_order).",
                type(broker_client).__name__,
            )
            return {
                "error": "no_supported_submit_method",
                "order_submitted": False,
                "broker_type": type(broker_client).__name__,
            }

        except Exception as e:
            self.log.error("❌ Broker submit error via %s: %s", type(broker_client).__name__, e, exc_info=True)
            return {"error": str(e), "order_submitted": False}

    def _extract_broker_order_id(self, broker_resp: Any) -> Optional[str]:
        if not broker_resp:
            return None
        if isinstance(broker_resp, dict):
            return str(broker_resp.get("id") or broker_resp.get("order_id") or "")
        oid = getattr(broker_resp, "id", None) or getattr(broker_resp, "client_order_id", None)
        return str(oid) if oid is not None else None

    def _extract_fill(self, broker_resp: Any) -> Dict[str, Any]:
        if not broker_resp:
            return {}
        if isinstance(broker_resp, dict):
            if "error" in broker_resp:
                return {}
            price = broker_resp.get("filled_avg_price") or broker_resp.get("filled_price")
            qty = broker_resp.get("filled_qty") or broker_resp.get("qty")
            return {
                "price": float(price) if price is not None else None,
                "qty": float(qty) if qty is not None else None,
            }
        price = getattr(broker_resp, "filled_avg_price", None)
        qty = getattr(broker_resp, "filled_qty", None)
        return {
            "price": float(price) if price is not None else None,
            "qty": float(qty) if qty is not None else None,
        }

    # ======================================================================
    # Risk Envelope Integration
    # ======================================================================

    def _apply_risk_envelope(
        self,
        order: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Run order through RiskEnvelopeController if available.

        Returns:
            (allowed, adjusted_order, risk_meta)
        """
        if not self.risk_ctrl:
            return True, order, {"status": "no_risk_controller"}

        try:
            if hasattr(self.risk_ctrl, "check_and_clamp"):
                out = self.risk_ctrl.check_and_clamp(order)
            elif hasattr(self.risk_ctrl, "evaluate"):
                portfolio = (
                    self.portfolio_provider()
                    if self.portfolio_provider is not None
                    else {
                        "equity": 0.0,
                        "positions": {},
                        "intraday_drawdown_pct": 0.0,
                    }
                )
                adjusted_order, meta = self.risk_ctrl.evaluate(portfolio, order)
                out = {"allowed": not bool(meta.get("hard_kill")), "order": adjusted_order, **meta}
            else:
                out = self.risk_ctrl(order)  # type: ignore[misc]

            if isinstance(out, dict):
                allowed = bool(out.get("allowed", True))
                adjusted = out.get("order", order)
                meta = {k: v for k, v in out.items() if k not in ("allowed", "order")}
            elif isinstance(out, (tuple, list)) and len(out) >= 2:
                allowed = bool(out[0])
                adjusted = out[1] or order
                meta = {"extra": out[2:]} if len(out) > 2 else {}
            else:
                allowed, adjusted, meta = True, order, {"status": "unknown_risk_api"}

            risk_reason = meta.get("reason")
            risk_heat = meta.get("risk_heat")
            risk_regime = meta.get("regime")
            clamped = bool(meta.get("clamped", False))
            hard_kill = bool(meta.get("hard_kill", False))

            try:
                heat_str = f"{float(risk_heat):.3f}" if risk_heat is not None else "n/a"
            except Exception:
                heat_str = "n/a"

            self.log.info(
                "RiskEnvelope: allowed=%s hard_kill=%s clamped=%s reason=%s heat=%s regime=%s",
                allowed,
                hard_kill,
                clamped,
                risk_reason,
                heat_str,
                risk_regime,
            )

            return allowed, adjusted, meta

        except Exception as e:
            self.log.warning("⚠️ RiskEnvelopeController failed; bypassing risk clamp: %s", e)
            return True, order, {"status": "risk_error", "error": str(e)}

    # ======================================================================
    # Prediction / Execution-Aware Gate
    # ======================================================================

    def _apply_prediction_gate(
        self,
        order: Dict[str, Any],
        extra: Dict[str, Any],
        out_meta: Dict[str, Any],
    ):
        if not self.slippage_predictor:
            return order, False, None

        try:
            sym = order["symbol"]
            side = order["side"]
            qty = float(order["qty"])
            order_type = order.get("order_type", "MARKET")
            limit_price = order.get("limit_price")

            features = self.slippage_predictor.build_features(
                symbol=sym,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price,
                extra=extra,
            )
            pred = self.slippage_predictor.predict(features)

            pred_slip = float(pred.get("slippage", 0.0))
            pred_latency_ms = float(pred.get("latency_ms", 0.0))
            fill_prob = float(pred.get("fill_probability", 1.0))

            out_meta.update(
                {
                    "predicted_slippage": pred_slip,
                    "predicted_latency_ms": pred_latency_ms,
                    "predicted_fill_probability": fill_prob,
                }
            )

            hard_block = False
            reason_parts = []

            if abs(pred_slip) > self.slippage_predictor.hard_slippage_limit:
                hard_block = True
                reason_parts.append("slippage_limit")

            if pred_latency_ms > self.slippage_predictor.hard_latency_limit_ms:
                hard_block = True
                reason_parts.append("latency_limit")

            if fill_prob < self.slippage_predictor.min_fill_probability:
                hard_block = True
                reason_parts.append("fill_probability")

            if hard_block:
                reason = " & ".join(reason_parts) or "prediction_hard_block"
                out_meta["blocked_by_prediction"] = True
                out_meta["block_reason"] = reason
                return order, True, reason

            clamped = False
            clamp_reason_parts = []
            new_order = dict(order)
            orig_qty = float(order.get("qty", 0.0))
            new_qty = orig_qty

            if abs(pred_slip) >= self.slippage_predictor.soft_slippage_limit:
                clamped = True
                clamp_reason_parts.append("soft_slippage")
                new_qty *= self.slippage_predictor.soft_slippage_qty_scale

            if pred_latency_ms >= self.slippage_predictor.soft_latency_limit_ms:
                clamped = True
                clamp_reason_parts.append("soft_latency")
                new_qty *= self.slippage_predictor.soft_latency_qty_scale

            if clamped and new_qty < orig_qty:
                new_order["qty"] = max(new_qty, 0.0)
                out_meta["clamped_by_prediction"] = True
                out_meta["clamp_reason"] = " & ".join(clamp_reason_parts)
                out_meta["original_qty"] = orig_qty
                out_meta["new_qty"] = new_order["qty"]

                self.log.info(
                    "Prediction clamp: %s %s qty %.4f → %.4f (%s)",
                    side,
                    sym,
                    orig_qty,
                    new_order["qty"],
                    out_meta["clamp_reason"],
                )
                return new_order, False, None

            return order, False, None

        except Exception as e:
            self.log.warning("⚠️ Prediction gate failed; continuing without clamp: %s", e)
            return order, False, None

    # ======================================================================
    # Trade journal helper
    # ======================================================================

    def _write_trade_journal_entry(
        self,
        *,
        ts_utc: str,
        order: Dict[str, Any],
        router_meta: Dict[str, Any],
        broker_name: Optional[str],
        broker_order_id: Optional[str],
        fill_price: Optional[float],
        latency_ms: int,
        status: str,
        error: Optional[str],
        journal_ctx: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> None:

        entry = {
            "ts_utc": ts_utc,
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "qty": order.get("qty"),
            "order_type": order.get("order_type"),
            "limit_price": order.get("limit_price"),
            "tag": order.get("tag"),
            "is_flattening": order.get("is_flattening"),
            "client_order_id": order.get("client_order_id"),
            "broker": broker_name,
            "broker_order_id": broker_order_id,
            "fill_price": fill_price,
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "router_meta": router_meta,
            "journal_ctx": journal_ctx,
            "extra": extra,
        }

        try:
            TradeJournal.append_entry(entry)
        except Exception:
            pass
