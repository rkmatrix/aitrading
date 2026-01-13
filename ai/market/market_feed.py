"""
ai/market/market_feed.py
------------------------

Phase 26+ MarketFeed with Guardrails (Phase 122.3)

Responsibilities
----------------
- Provide latest ticks/bars for symbols.
- Track staleness and gaps.
- Apply GuardrailRuntimeHandler to detect stale / broken feed.

This module assumes you have a data adapter that implements:
    get_latest_quote(symbol) -> dict with at least:
        { "symbol": str, "bid": float, "ask": float, "last": float,
          "ts": float (unix timestamp) }

If you already have a custom adapter, wire it in via the `data_source` callable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from ai.safety.auto_guardrails import GuardrailRuntimeHandler, GuardrailDecision

logger = logging.getLogger(__name__)


class MarketFeed:
    """
    Thin wrapper around an underlying data source, with guardrails.
    """

    def __init__(
        self,
        *,
        data_source: Callable[[str], Dict[str, Any]],
        guardrails: Optional[GuardrailRuntimeHandler] = None,
        max_staleness_sec: float = 5.0,
    ) -> None:
        """
        data_source: callable like `lambda symbol: {...quote...}`
        """
        self.log = logging.getLogger("MarketFeed")
        self.data_source = data_source
        self.guardrails = guardrails
        self.max_staleness_sec = float(max_staleness_sec)

        self.log.info(
            "MarketFeed initialized (max_staleness_sec=%.1f)", self.max_staleness_sec
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_latest_tick(self, symbol: str) -> Dict[str, Any]:
        """
        Return latest tick dict, or a stub if guardrails block.
        Expected fields:
            symbol, bid, ask, last, ts
        """
        tick = self._safe_get_quote(symbol)
        now = time.time()

        if not tick:
            return {"symbol": symbol, "stale": True, "error": "no_data"}

        ts = tick.get("ts") or tick.get("timestamp") or now
        try:
            ts = float(ts)
        except Exception:
            ts = now

        staleness = max(0.0, now - ts)
        tick["staleness_sec"] = staleness

        # Basic hard safety even if guardrails are off
        if staleness > self.max_staleness_sec and self.guardrails is None:
            self.log.warning(
                "⏳ Market data stale for %s: staleness=%.2fs (>%.2fs)",
                symbol,
                staleness,
                self.max_staleness_sec,
            )

        if self.guardrails is not None:
            decision: GuardrailDecision = self.guardrails.check(
                event="tick",
                metrics={"staleness_sec": staleness},
                context={"symbol": symbol},
            )

            if decision.is_blocked:
                self.log.error(
                    "⛔ Guardrails blocked market tick for %s (staleness=%.2fs): %s",
                    symbol,
                    staleness,
                    "; ".join(decision.reasons),
                )
                return {
                    "symbol": symbol,
                    "stale": True,
                    "staleness_sec": staleness,
                    "blocked_by_guardrails": True,
                    "guardrail_reasons": decision.reasons,
                }

        return tick

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _safe_get_quote(self, symbol: str) -> Dict[str, Any]:
        try:
            quote = self.data_source(symbol)
            if not isinstance(quote, dict):
                self.log.warning("Data source returned non-dict for %s: %r", symbol, quote)
                return {}
            if "symbol" not in quote:
                quote["symbol"] = symbol
            return quote
        except Exception as e:
            self.log.error("MarketFeed data_source error for %s: %s", symbol, e, exc_info=True)
            return {"symbol": symbol, "error": str(e), "stale": True}
