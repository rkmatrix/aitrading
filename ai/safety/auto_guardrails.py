# ai/safety/auto_guardrails.py
"""
Phase 122.3 – Auto Guardrails

Runtime safety guardrails that sit between decision logic (RL policy,
PortfolioBrain, PositionSizer, etc.) and execution (SmartOrderRouter).

Key components
--------------
- GuardrailLoader
    Loads YAML config + Phase 122 failure logs and builds an in-memory
    rule set (static + auto-derived).

- GuardrailEngine
    Evaluates incoming metrics for a (component, event) pair and returns
    a GuardrailDecision (allow, clamp, block, downgrade, failover).

- GuardrailRuntimeHandler
    Lightweight facade used by runtime components (Phase 26 realtime
    loop, SmartOrderRouter, etc.) to check and log guardrail decisions.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import yaml

logger = logging.getLogger(__name__)

ActionType = Literal["allow", "clamp", "block", "downgrade", "failover"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GuardrailRule:
    id: str
    component: str
    event: Optional[str] = None  # None = any event
    metric: Optional[str] = None  # None = aggregate/context-only
    op: str = "<="               # "<", "<=", ">", ">=", "==", "!="
    threshold: Optional[float] = None
    action: ActionType = "block"
    severity: str = "WARNING"
    message: str = ""
    clamp_to: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailDecision:
    action: ActionType
    reasons: List[str] = field(default_factory=list)
    rule_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.action == "allow"

    @property
    def is_blocked(self) -> bool:
        return self.action == "block"

    @property
    def is_clamp(self) -> bool:
        return self.action == "clamp"

    @property
    def is_downgrade(self) -> bool:
        return self.action == "downgrade"

    @property
    def is_failover(self) -> bool:
        return self.action == "failover"


# ---------------------------------------------------------------------------
# Guardrail Engine
# ---------------------------------------------------------------------------


class GuardrailEngine:
    """
    Evaluates GuardrailRule objects for a given component/event.

    - metrics: dict of numeric/boolean values from caller.
    - context: arbitrary metadata (symbol, side, account, etc.).
    """

    def __init__(
        self,
        *,
        rules: Iterable[GuardrailRule],
        mode: str = "ENABLED",
        global_limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rules: List[GuardrailRule] = list(rules)
        self.mode = mode.upper()
        self.global_limits = global_limits or {}

    # ---- public API --------------------------------------------------------

    def evaluate(
        self,
        component: str,
        event: Optional[str],
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailDecision:
        if self.mode == "DISABLED":
            return GuardrailDecision(action="allow")

        context = context or {}

        # 1) Global coarse-grained safety
        decision = self._check_global_limits(component, event, metrics, context)
        if decision.action in ("block", "failover"):
            return decision

        matching_rules = self._matching_rules(component, event)

        if not matching_rules:
            if self.mode == "DRY_RUN":
                logger.debug(
                    "Guardrails (DRY_RUN): no rules for %s/%s metrics=%s context=%s",
                    component,
                    event,
                    metrics,
                    context,
                )
            return GuardrailDecision(action="allow")

        final_action: ActionType = "allow"
        reasons: List[str] = []
        rule_ids: List[str] = []
        adjusted_metrics = dict(metrics)

        for rule in matching_rules:
            if not rule.metric:
                # Context-only rule; not implemented here
                continue

            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]
            if not self._is_number(value):
                continue

            if self._compare(float(value), rule.op, rule.threshold):
                reasons.append(rule.message or f"Rule {rule.id} triggered.")
                rule_ids.append(rule.id)

                if rule.action == "clamp":
                    if rule.clamp_to is not None:
                        adjusted_metrics[rule.metric] = min(
                            float(value),
                            float(rule.clamp_to),
                        )
                    final_action = _max_severity_action(final_action, "clamp")
                else:
                    final_action = _max_severity_action(final_action, rule.action)

        if self.mode == "DRY_RUN":
            if reasons:
                logger.warning(
                    "Guardrails (DRY_RUN) would take action=%s on %s/%s: %s",
                    final_action,
                    component,
                    event,
                    "; ".join(reasons),
                )
            return GuardrailDecision(
                action="allow",
                reasons=reasons,
                rule_ids=rule_ids,
                metrics=adjusted_metrics,
            )

        return GuardrailDecision(
            action=final_action,
            reasons=reasons,
            rule_ids=rule_ids,
            metrics=adjusted_metrics,
        )

    # ---- internals ---------------------------------------------------------

    def _matching_rules(self, component: str, event: Optional[str]) -> List[GuardrailRule]:
        out: List[GuardrailRule] = []
        for r in self.rules:
            if r.component != component:
                continue
            if r.event is not None and event is not None and r.event != event:
                continue
            out.append(r)
        return out

    def _check_global_limits(
        self,
        component: str,
        event: Optional[str],
        metrics: Dict[str, Any],
        context: Dict[str, Any],
    ) -> GuardrailDecision:
        if not self.global_limits:
            return GuardrailDecision(action="allow")

        reasons: List[str] = []
        rule_ids: List[str] = []
        action: ActionType = "allow"

        gl = self.global_limits

        # Equity drawdown
        dd_key = "equity_drawdown_pct"
        if dd_key in metrics and self._is_number(metrics[dd_key]):
            max_dd = gl.get("max_equity_drawdown_pct")
            if max_dd is not None and float(metrics[dd_key]) < -abs(float(max_dd)):
                reasons.append(
                    f"Equity drawdown {metrics[dd_key]:.2f}% < -{max_dd:.2f}% (global limit)."
                )
                rule_ids.append("global_equity_drawdown")
                action = _max_severity_action(action, "block")

        # Daily loss
        pnl_key = "daily_pnl_pct"
        if pnl_key in metrics and self._is_number(metrics[pnl_key]):
            max_loss = gl.get("max_daily_loss_pct")
            if max_loss is not None and float(metrics[pnl_key]) < -abs(float(max_loss)):
                reasons.append(
                    f"Daily PnL {metrics[pnl_key]:.2f}% < -{max_loss:.2f}% (global limit)."
                )
                rule_ids.append("global_daily_loss")
                action = _max_severity_action(action, "block")

        # Position count
        pos_key = "open_positions"
        if pos_key in metrics and self._is_number(metrics[pos_key]):
            max_pos = gl.get("max_open_positions")
            if max_pos is not None and float(metrics[pos_key]) > float(max_pos):
                reasons.append(
                    f"Open positions {metrics[pos_key]} > {max_pos} (global limit)."
                )
                rule_ids.append("global_open_positions")
                action = _max_severity_action(action, "block")

        if action == "allow":
            return GuardrailDecision(action="allow")

        if self.mode == "DRY_RUN":
            logger.warning(
                "Guardrails (DRY_RUN) global would block %s/%s: %s",
                component,
                event,
                "; ".join(reasons),
            )
            return GuardrailDecision(action="allow", reasons=reasons, rule_ids=rule_ids)

        return GuardrailDecision(action=action, reasons=reasons, rule_ids=rule_ids)

    @staticmethod
    def _is_number(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _compare(value: float, op: str, threshold: Optional[float]) -> bool:
        if threshold is None:
            return False
        if op == "<":
            return value < threshold
        if op == "<=":
            return value <= threshold
        if op == ">":
            return value > threshold
        if op == ">=":
            return value >= threshold
        if op == "==":
            return value == threshold
        if op == "!=":
            return value != threshold
        return False


def _max_severity_action(current: ActionType, new: ActionType) -> ActionType:
    """
    Combine two actions, returning the more severe one.
    Severity order:
        allow < clamp < downgrade < failover < block
    """
    severity_order = {
        "allow": 0,
        "clamp": 1,
        "downgrade": 2,
        "failover": 3,
        "block": 4,
    }
    return new if severity_order[new] > severity_order[current] else current


# ---------------------------------------------------------------------------
# Guardrail Loader – config + Phase 122 logs → GuardrailEngine
# ---------------------------------------------------------------------------


class GuardrailLoader:
    """
    Loads YAML guardrail configuration and, optionally, derives additional
    rules from Phase 122 failure harness logs.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.raw_cfg: Dict[str, Any] = {}
        self.rules: List[GuardrailRule] = []
        self.mode: str = "ENABLED"
        self.global_limits: Dict[str, Any] = {}

    def build_engine(self) -> GuardrailEngine:
        self._load_yaml()
        self._load_static_rules()
        self._maybe_derive_auto_rules_from_failures()

        return GuardrailEngine(
            rules=self.rules,
            mode=self.raw_cfg.get("mode", "ENABLED"),
            global_limits=self.raw_cfg.get("global_limits", {}),
        )

    def _load_yaml(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Guardrail config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            self.raw_cfg = yaml.safe_load(f) or {}
        self.mode = self.raw_cfg.get("mode", "ENABLED")
        self.global_limits = self.raw_cfg.get("global_limits", {})

        log_level = getattr(
            logging,
            self.raw_cfg.get("log_level", "INFO").upper(),
            logging.INFO,
        )
        logger.setLevel(log_level)

    def _load_static_rules(self) -> None:
        rules_cfg = self.raw_cfg.get("rules", []) or []
        for rc in rules_cfg:
            rule = GuardrailRule(
                id=str(rc.get("id", f"rule_{len(self.rules)+1}")),
                component=str(rc.get("component")),
                event=rc.get("event"),
                metric=rc.get("metric"),
                op=str(rc.get("op", "<=")),
                threshold=rc.get("threshold"),
                action=str(rc.get("action", "block")),
                severity=str(rc.get("severity", "WARNING")),
                message=str(rc.get("message", "")),
                clamp_to=rc.get("clamp_to"),
                extra={
                    k: v
                    for k, v in rc.items()
                    if k
                    not in {
                        "id",
                        "component",
                        "event",
                        "metric",
                        "op",
                        "threshold",
                        "action",
                        "severity",
                        "message",
                        "clamp_to",
                    }
                },
            )
            self.rules.append(rule)

    def _maybe_derive_auto_rules_from_failures(self) -> None:
        auto_cfg = self.raw_cfg.get("auto_rules", {}) or {}
        if not auto_cfg.get("enabled", False):
            return

        logs_cfg = self.raw_cfg.get("logs", {}) or {}
        if not logs_cfg.get("learn_from_failures", False):
            return

        failure_path_str = logs_cfg.get("failure_log_path")
        if not failure_path_str:
            return

        failure_path = Path(failure_path_str)
        if not failure_path.exists():
            logger.info(
                "GuardrailLoader: no failure log found at %s; skipping auto-rules",
                failure_path,
            )
            return

        min_occ = int(logs_cfg.get("min_occurrences_for_rule", 3))
        safety_margin_pct = float(auto_cfg.get("safety_margin_pct", 5.0))
        default_action = auto_cfg.get("default_action", "block")
        default_op = auto_cfg.get("default_op", "<=")

        stats: Dict[tuple, List[float]] = {}

        cutoff_ts = None
        lookback_days = int(logs_cfg.get("lookback_days", 14))
        if lookback_days > 0:
            cutoff_ts = time.time() - lookback_days * 86400

        with failure_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = rec.get("ts")
                if cutoff_ts is not None and isinstance(ts, (float, int)) and ts < cutoff_ts:
                    continue

                component = rec.get("component")
                event = rec.get("event")
                metrics = rec.get("metrics") or {}
                failed_metric = rec.get("failed_metric")

                if not component or not isinstance(metrics, dict):
                    continue

                if failed_metric and failed_metric in metrics:
                    metric_keys = [failed_metric]
                else:
                    metric_keys = [
                        k
                        for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not math.isnan(v)
                    ]

                for mk in metric_keys:
                    val = metrics.get(mk)
                    if not isinstance(val, (int, float)) or math.isnan(val):
                        continue
                    key = (component, event, mk)
                    stats.setdefault(key, []).append(float(val))

        for (component, event, metric), values in stats.items():
            if len(values) < min_occ:
                continue

            max_fail = max(values)
            if not math.isfinite(max_fail):
                continue

            threshold = max_fail * (1.0 - safety_margin_pct / 100.0)
            rule_id = f"auto_{component}_{event or 'any'}_{metric}"

            logger.info(
                "GuardrailLoader: deriving auto-rule %s for %s/%s/%s: samples=%d threshold=%.2f",
                rule_id,
                component,
                event,
                metric,
                len(values),
                threshold,
            )

            self.rules.append(
                GuardrailRule(
                    id=rule_id,
                    component=component,
                    event=event,
                    metric=metric,
                    op=default_op,
                    threshold=threshold,
                    action=default_action,
                    severity="ERROR",
                    message=(
                        f"Auto rule from Phase 122 failures: {metric} must be "
                        f"{default_op} {threshold:.2f}"
                    ),
                )
            )


# ---------------------------------------------------------------------------
# Guardrail Runtime Handler – used by runtime components
# ---------------------------------------------------------------------------


class GuardrailRuntimeHandler:
    """
    Thin wrapper around GuardrailEngine that attaches a default component
    name and logs decisions in a consistent way.
    """

    def __init__(
        self,
        engine: GuardrailEngine,
        *,
        component: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.engine = engine
        self.component = component
        self.log = logger_ or logging.getLogger(
            f"Guardrails[{component or 'generic'}]"
        )

    def for_component(self, component: str) -> "GuardrailRuntimeHandler":
        return GuardrailRuntimeHandler(self.engine, component=component, logger_=self.log)

    def check(
        self,
        event: Optional[str],
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailDecision:
        component = self.component or metrics.get("component") or "unknown"
        decision = self.engine.evaluate(component, event, metrics, context)

        if decision.action != "allow" and decision.reasons:
            self.log.warning(
                "Guardrail %s/%s → action=%s rules=%s reasons=%s metrics=%s",
                component,
                event,
                decision.action,
                ",".join(decision.rule_ids),
                "; ".join(decision.reasons),
                metrics,
            )
        elif self.engine.mode == "DRY_RUN" and decision.reasons:
            self.log.debug(
                "(DRY_RUN) Guardrail %s/%s would take action=%s rules=%s reasons=%s metrics=%s",
                component,
                event,
                decision.action,
                ",".join(decision.rule_ids),
                "; ".join(decision.reasons),
                metrics,
            )
        return decision

    def guard_call(
        self,
        event: str,
        metrics: Dict[str, Any],
        call_fn: Callable[[], Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        on_block: Optional[Callable[[GuardrailDecision], Any]] = None,
    ) -> Any:
        """
        Wrapper to guard execution of a side-effecting call.
        """
        decision = self.check(event, metrics, context)

        if decision.is_blocked:
            if on_block:
                return on_block(decision)
            return None

        return call_fn()
