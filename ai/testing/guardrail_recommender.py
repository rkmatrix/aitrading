# ai/testing/guardrail_recommender.py
# Phase 122.3 – Auto-Guardrail Recommender

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GuardrailRecommender")


# ------------------------------------------------------
# Data structures
# ------------------------------------------------------


@dataclass
class GuardrailSuggestion:
    scenario_name: str
    worst_classification: str
    tags: List[str]

    module_path: str
    function_name: str

    summary: str
    rationale: str
    patch_snippet: str


# ------------------------------------------------------
# Loading diagnostics
# ------------------------------------------------------


def load_diagnostics(json_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load Phase 122.2 diagnostic JSON.

    Expected format: list of objects with keys:
      - scenario_name
      - worst_classification
      - tags
      - ...
    """
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(
            f"Diagnostic JSON not found at {json_path}.\n"
            f"Run Phase 122.2 first (phase122_diagnostic_analyzer)."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Diagnostic JSON at {json_path} is not a list")

    logger.info("Loaded %d diagnostic scenario entries from %s", len(data), json_path)
    return data


# ------------------------------------------------------
# Mapping from scenario → guardrail recipe
# ------------------------------------------------------


def _matches(name: str, tags: List[str], needle: str) -> bool:
    return needle.lower() in name.lower()


def build_guardrail_for_scenario(diag: Dict[str, Any]) -> Optional[GuardrailSuggestion]:
    """
    Map a single diagnostic entry to a GuardrailSuggestion, if applicable.

    Only CRITICAL_CRASH scenarios should reach here, but we don't strictly rely on that.
    """
    name: str = diag.get("scenario_name", "")
    worst: str = diag.get("worst_classification", "UNKNOWN")
    tags: List[str] = list(diag.get("tags", []))

    # Default fallback: None (we only generate for known patterns)
    # You can extend this mapping over time.
    # --------------------------------------------------
    # BROKER GUARDRAILS
    # --------------------------------------------------
    if _matches(name, tags, "Broker outage") or (
        "broker" in tags and "critical" in tags
    ):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.execution.broker_alpaca_live",
            function_name="AlpacaClient.submit_order",
            summary="Wrap Alpaca submit_order with try/except and convert hard failures "
                    "into a safe, logged error object so the rest of the system can continue.",
            rationale=(
                "When the underlying Alpaca client raises (network outage, rate limit, etc.), "
                "your live loop crashes. Instead, catch exceptions, log, optionally notify via "
                "Telegram, and return a sentinel failure object that the execution agent can inspect."
            ),
            patch_snippet="""\
# Example inside ai/execution/broker_alpaca_live.py

class AlpacaClient:
    # ... existing __init__ ...

    def submit_order(self, symbol: str, side: str, qty: float, **kwargs):
        try:
            # existing call to Alpaca SDK here, e.g.:
            #   order = self._client.submit_order(symbol=symbol, side=side, qty=qty, **kwargs)
            order = self._client.submit_order(symbol=symbol, side=side, qty=qty, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.log.error("Broker submit_order failed for %s %s x%.2f: %s", side, symbol, qty, e)
            try:
                from tools.telegram_alerts import notify
                notify(f"🚨 Broker submit_order failed for {side} {symbol} x{qty}: {e}")
            except Exception:
                self.log.exception("Failed to send Telegram notification for broker error")

            # Return a structured failure object instead of raising
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "error_message": str(e),
                "symbol": symbol,
                "side": side,
                "qty": qty,
            }

        # Normalize the response (dict) before returning
        return {
            "status": "ok",
            "id": getattr(order, "id", None),
            "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
            "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
            "raw": order,
        }
""",
        )

    if _matches(name, tags, "malformed response"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.execution.broker_alpaca_live",
            function_name="AlpacaClient.submit_order",
            summary="Validate broker response structure before use; if malformed, "
                    "log and convert to a safe failure object.",
            rationale=(
                "Your downstream code assumes a dict-like order object with certain keys. "
                "When the response is malformed, it crashes when accessing fields. "
                "Add a small validator that checks type & keys and degrades gracefully."
            ),
            patch_snippet="""\
# Example validation helper in ai/execution/broker_alpaca_live.py

class AlpacaClient:
    # ...

    def _normalize_order_response(self, order) -> dict:
        # Coerce to dict if needed (SDK object → dict)
        if hasattr(order, "id"):
            # Example conversion; customize for your SDK
            order_dict = {
                "id": getattr(order, "id", None),
                "filled_qty": getattr(order, "filled_qty", 0),
                "filled_avg_price": getattr(order, "filled_avg_price", 0),
            }
        elif isinstance(order, dict):
            order_dict = dict(order)
        else:
            self.log.error("Malformed broker response (not dict / not SDK object): %r", order)
            return {
                "status": "error",
                "error_type": "MalformedResponse",
                "error_message": f"Unexpected order response type: {type(order)}",
                "raw": repr(order),
            }

        # Basic sanity checks
        if "id" not in order_dict:
            self.log.error("Malformed broker response missing 'id': %r", order_dict)
            order_dict.setdefault("status", "error")
            order_dict.setdefault("error_type", "MalformedResponse")
            return order_dict

        order_dict.setdefault("status", "ok")
        return order_dict

    def submit_order(self, symbol: str, side: str, qty: float, **kwargs):
        try:
            order = self._client.submit_order(symbol=symbol, side=side, qty=qty, **kwargs)
        except Exception as e:
            # (see previous snippet for exception handling)
            ...
        return self._normalize_order_response(order)
""",
        )

    if _matches(name, tags, "negative fill") or _matches(name, tags, "negative/zero prices"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.execution.broker_alpaca_live",
            function_name="AlpacaClient.submit_order",
            summary="Clamp and validate numeric fields (filled_qty, filled_avg_price). "
                    "If invalid values are detected, flag the order as error instead of crashing logic downstream.",
            rationale=(
                "Downstream PnL and position logic may assume positive quantities/prices; "
                "guard against negative or zero values and record them as broker anomalies."
            ),
            patch_snippet="""\
# Example numeric validation inside _normalize_order_response

    def _normalize_order_response(self, order) -> dict:
        # ... as before, build order_dict ...
        order_dict = {...}  # build from SDK or dict

        qty = float(order_dict.get("filled_qty", 0) or 0)
        price = float(order_dict.get("filled_avg_price", 0) or 0)

        if qty < 0 or price <= 0:
            self.log.error("Invalid broker fills (qty=%.4f, price=%.4f): %r", qty, price, order_dict)
            return {
                "status": "error",
                "error_type": "InvalidFillValues",
                "error_message": f"qty={qty}, price={price}",
                "raw": order_dict,
            }

        order_dict["filled_qty"] = qty
        order_dict["filled_avg_price"] = price
        order_dict.setdefault("status", "ok")
        return order_dict
""",
        )

    # --------------------------------------------------
    # DATA / MARKET FEED GUARDRAILS
    # --------------------------------------------------
    if _matches(name, tags, "Market data returns None") or ("data" in tags and "None" in name):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.data.market_data",
            function_name="AlpacaDataClient.get_last_quote",
            summary="When data provider returns None, fall back to previous quote or "
                    "mark symbol as temporarily unavailable instead of crashing.",
            rationale=(
                "Your signal/position logic likely assumes a quote dict. If None bubbles up, "
                "you get attribute errors. Guard by returning a stub / cached quote or "
                "explicit 'unavailable' marker that higher layers can interpret."
            ),
            patch_snippet="""\
# Example inside ai/data/market_data.py

class AlpacaDataClient:
    # ...

    def get_last_quote(self, symbol: str) -> dict | None:
        try:
            q = self._client.get_last_quote(symbol)
        except Exception as e:  # noqa: BLE001
            self.log.error("Market data error for %s: %s", symbol, e)
            return {"status": "error", "symbol": symbol, "reason": str(e)}

        if q is None:
            self.log.warning("No quote returned for %s (None). Marking as unavailable.", symbol)
            return {"status": "unavailable", "symbol": symbol}

        # Convert SDK quote object → dict and tag status
        quote = {
            "status": "ok",
            "symbol": symbol,
            "bid": float(getattr(q, "bid", 0) or 0),
            "ask": float(getattr(q, "ask", 0) or 0),
            "timestamp": getattr(q, "timestamp", None),
        }
        return quote
""",
        )

    if _matches(name, tags, "latency spike") or ("latency" in tags and "data" in tags):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.data.market_data",
            function_name="AlpacaDataClient.get_last_quote",
            summary="Wrap price fetch in a soft timeout measurement and, if too slow, "
                    "return a 'stale' or 'timeout' marker instead of blocking the whole loop.",
            rationale=(
                "When data fetch takes 5–10s, your realtime loop stalls. You should log and "
                "degrade gracefully (e.g., skip this tick) instead of freezing execution."
            ),
            patch_snippet="""\
import time

class AlpacaDataClient:
    # ...

    def get_last_quote(self, symbol: str) -> dict:
        start = time.time()
        q = self._client.get_last_quote(symbol)
        elapsed = time.time() - start

        if elapsed > 3.0:  # soft timeout threshold in seconds
            self.log.warning("Slow quote fetch for %s: %.3fs", symbol, elapsed)

        # ... normalize as in previous snippet ...
        ...
""",
        )

    # --------------------------------------------------
    # RISK ENVELOPE GUARDRAILS
    # --------------------------------------------------
    if _matches(name, tags, "RiskEnvelope clamps everything") or ("risk" in tags and "block" in name.lower()):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.risk.risk_envelope",
            function_name="RiskEnvelopeController.block_order",
            summary="Ensure block_order has strong logging and a safety escape hatch "
                    "if it suddenly starts blocking 100% of orders.",
            rationale=(
                "If RiskEnvelope misconfigures and blocks all orders, the bot looks 'dead'. "
                "Monitor block rate and trigger alerts if everything is rejected."
            ),
            patch_snippet="""\
# Inside ai/risk/risk_envelope.py

class RiskEnvelopeController:
    def __init__(self, *args, **kwargs):
        # ...
        self._recent_block_count = 0
        self._recent_total_count = 0

    def block_order(self, order) -> bool:
        # existing checks...
        blocked = self._internal_block_logic(order)

        # Track basic metrics
        self._recent_total_count += 1
        if blocked:
            self._recent_block_count += 1

        # If everything is suddenly blocked, warn loudly
        if self._recent_total_count >= 10:
            block_rate = self._recent_block_count / max(self._recent_total_count, 1)
            if block_rate >= 0.9:
                self.log.error("RiskEnvelope blocking %.0f%% of orders (recent=%d/%d)",
                               block_rate * 100, self._recent_block_count, self._recent_total_count)
                try:
                    from tools.telegram_alerts import notify
                    notify(f\"🚨 RiskEnvelope blocking {block_rate*100:.0f}% of orders. Check config.\")
                except Exception:
                    self.log.exception("Failed to send RiskEnvelope alert")

            # Reset window
            self._recent_block_count = 0
            self._recent_total_count = 0

        return blocked
""",
        )

    if _matches(name, tags, "RiskEnvelope raises ValueError"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.risk.risk_envelope",
            function_name="RiskEnvelopeController.clamp",
            summary="Wrap clamp logic with validation of min/max bounds and "
                    "catch ValueError to return safe defaults.",
            rationale=(
                "If clamp receives invalid inputs and raises, you crash right when enforcing risk. "
                "Better to log the bad inputs and fall back to original order size or a strongly "
                "clamped size than to kill the loop."
            ),
            patch_snippet="""\
class RiskEnvelopeController:
    # ...

    def clamp(self, qty: float, symbol: str | None = None) -> float:
        try:
            min_q, max_q = self._compute_bounds(symbol)
        except Exception as e:  # noqa: BLE001
            self.log.error("RiskEnvelope clamp failed for %s qty=%.4f: %s", symbol, qty, e)
            return qty  # fail-open or choose a conservative clamp

        if min_q > max_q:
            self.log.error("Invalid clamp bounds for %s: min=%.4f > max=%.4f", symbol, min_q, max_q)
            # choose a safe behavior (e.g., no trade)
            return 0.0

        clamped = max(min_q, min(max_q, qty))
        return clamped
""",
        )

    # --------------------------------------------------
    # POLICY GUARDRAILS
    # --------------------------------------------------
    if _matches(name, tags, "Policy returns invalid action"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.policy.multi_policy_supervisor",
            function_name="MultiPolicySupervisor.predict",
            summary="Validate policy action against action space; if invalid, "
                    "fallback to neutral action or previous safe action.",
            rationale=(
                "RL policies occasionally output NaN or out-of-bounds actions. "
                "You should guard the final output of MultiPolicySupervisor before sending to the env/executor."
            ),
            patch_snippet="""\
# Inside ai/policy/multi_policy_supervisor.py

class MultiPolicySupervisor:
    # ...

    def predict(self, obs, deterministic: bool = True):
        action = self._active_policy.predict(obs, deterministic=deterministic)
        action = self._postprocess_action(action)
        return action

    def _postprocess_action(self, action):
        import numpy as np

        arr = np.array(action, dtype=float)
        if not np.all(np.isfinite(arr)):
            self.log.error("Policy produced non-finite action: %r", action)
            # fallback to zeros or previous valid action
            return np.zeros_like(arr)

        if hasattr(self, "action_space"):
            low, high = self.action_space.low, self.action_space.high
            if arr.shape == low.shape:
                clipped = np.clip(arr, low, high)
                if not np.allclose(clipped, arr):
                    self.log.warning("Clipping out-of-bounds action %r → %r", arr, clipped)
                return clipped

        return arr
""",
        )

    if _matches(name, tags, "Policy raises exception"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.policy.multi_policy_supervisor",
            function_name="MultiPolicySupervisor.predict",
            summary="Wrap internal policy.predict with try/except, log, and return neutral action on failure.",
            rationale=(
                "When a policy crashes mid-run, the whole loop dies. Instead, treat it as a transient "
                "policy failure and fall back to a neutral (hold/no-trade) action while alerting."
            ),
            patch_snippet="""\
class MultiPolicySupervisor:
    # ...

    def predict(self, obs, deterministic: bool = True):
        try:
            action = self._active_policy.predict(obs, deterministic=deterministic)
        except Exception as e:  # noqa: BLE001
            self.log.error("Policy.predict crashed: %s", e)
            try:
                from tools.telegram_alerts import notify
                notify(f"🚨 Policy.predict crashed: {e}")
            except Exception:
                self.log.exception("Failed to send policy crash alert")

            # Fallback to neutral / no-op action
            if hasattr(self, "action_space"):
                return self.action_space.sample() * 0  # zeroed sample
            return 0
        return self._postprocess_action(action)
""",
        )

    # --------------------------------------------------
    # PORTFOLIO / POSITION GUARDRAILS
    # --------------------------------------------------
    if _matches(name, tags, "PortfolioBrain invalid weights"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.allocators.portfolio_brain",
            function_name="PortfolioBrain.compute_weights",
            summary="Normalize weights and validate sum; if out of range or NaN, "
                    "fall back to previous weights or equal-weight portfolio.",
            rationale=(
                "Upstream signals might generate crazy weights. A simple normalization "
                "and validation layer turns these into safe allocations instead of "
                "crashing or over-leveraging."
            ),
            patch_snippet="""\
class PortfolioBrain:
    # ...

    def compute_weights(self, inputs) -> dict:
        raw = self._compute_raw_weights(inputs)  # existing logic
        # Ensure dict
        weights = dict(raw)

        total = float(sum(weights.values()))
        if not weights or not (0.0 < total <= 1.5):
            self.log.error("Invalid portfolio weights sum=%.4f: %r", total, weights)
            # Fallback: equal weight among symbols
            n = max(len(weights), 1)
            eq = 1.0 / n
            return {k: eq for k in weights.keys()}

        # Normalize to sum=1.0
        return {k: v / total for k, v in weights.items()}
""",
        )

    if _matches(name, tags, "negative shares"):
        return GuardrailSuggestion(
            scenario_name=name,
            worst_classification=worst,
            tags=tags,
            module_path="ai.allocators.position_allocator",
            function_name="PositionAllocator.compute_sizes",
            summary="Clamp negative share sizes to zero or treat as a short only if explicitly allowed.",
            rationale=(
                "Unexpected negative shares can flip direction or cause risk explosion. "
                "Enforce a clear rule: either allow controlled shorts or clamp negatives to zero."
            ),
            patch_snippet="""\
class PositionAllocator:
    # ...

    def compute_sizes(self, weights: dict, prices: dict) -> dict:
        sizes = self._existing_compute_logic(weights, prices)

        cleaned = {}
        for sym, info in sizes.items():
            sh = float(info.get("shares", 0) or 0)
            if sh < 0:
                self.log.error("Negative position for %s detected (%.4f). Clamping to 0.", sym, sh)
                sh = 0.0
            cleaned[sym] = {**info, "shares": sh}
        return cleaned
""",
        )

    # No known mapping → skip
    return None


def build_guardrails_from_diagnostics(diags: List[Dict[str, Any]]) -> List[GuardrailSuggestion]:
    """
    Build guardrail suggestions for all CRITICAL_CRASH scenarios we recognize.
    """
    suggestions: List[GuardrailSuggestion] = []

    for d in diags:
        if d.get("worst_classification") != "CRITICAL_CRASH":
            continue
        s = build_guardrail_for_scenario(d)
        if s:
            suggestions.append(s)

    # sort by module then function
    suggestions.sort(key=lambda g: (g.module_path, g.function_name, g.scenario_name))
    logger.info("Generated %d guardrail suggestions.", len(suggestions))
    return suggestions


# ------------------------------------------------------
# Output writers
# ------------------------------------------------------


def write_guardrail_reports(
    suggestions: List[GuardrailSuggestion],
    report_dir: str | Path,
) -> Dict[str, str]:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "phase122_guardrails.json"
    md_path = report_dir / "phase122_guardrails.md"

    payload = [asdict(s) for s in suggestions]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown
    lines: List[str] = []
    lines.append("# Phase 122.3 – Auto-Guardrail Recommendations\n")

    if not suggestions:
        lines.append("_No CRITICAL_CRASH scenarios detected or no known mappings. Nothing to recommend._")
    else:
        for s in suggestions:
            lines.append(f"## {s.scenario_name}")
            lines.append(f"- Classification: **{s.worst_classification}**")
            if s.tags:
                lines.append(f"- Tags: `{', '.join(s.tags)}`")
            lines.append(f"- Target: `{s.module_path}.{s.function_name}()`")
            lines.append(f"- Summary: {s.summary}")
            lines.append(f"- Rationale: {s.rationale}")
            lines.append("")
            lines.append("```python")
            lines.append(s.patch_snippet.rstrip())
            lines.append("```")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Guardrail reports written:")
    logger.info("  JSON → %s", json_path)
    logger.info("  MD   → %s", md_path)
    return {"json": str(json_path), "md": str(md_path)}
