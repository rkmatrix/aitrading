"""
ai/execution/execution_alpha.py

Phase 80 — Execution Alpha Engine (ledger-aware)

This version understands your event-based order_ledger.jsonl format:

Events:
  • type = "submitted"  → initial order (symbol, side, qty, signal_price, ts_created)
  • type = "risk_check" → post-risk order snapshot
  • type = "reroute"    → broker + route_meta.ts_routed
  • type = "fill"       → broker + fill_price + _ts (fill time)
  • type = "complete"   → final summary (broker, fill_price, slippage, latency_ms, route_meta.ts_routed, _ts)

The engine:
  1. Ingests all events per order_id
  2. Reconstructs per-order execution state (symbol, side, broker, signal_price, fill_price, latency, etc.)
  3. Aggregates per (symbol, side, broker) bucket
  4. Produces slippage / latency / fill-ratio / execution score stats
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


def _parse_ts(val: Any) -> Optional[datetime]:
    """
    Best-effort timestamp parser.
    Accepts ISO strings, epoch seconds, epoch ms, or datetime.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        # Heuristic: if very large, treat as ms
        if val > 10_000_000_000:
            return datetime.utcfromtimestamp(val / 1000.0)
        return datetime.utcfromtimestamp(val)
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


@dataclass
class ExecutionBucketStats:
    """
    Aggregated stats for (symbol, side, broker).
    """

    count: int = 0
    filled_count: int = 0

    slippage_bps: List[float] = field(default_factory=list)
    slippage_abs: List[float] = field(default_factory=list)
    latency_sec: List[float] = field(default_factory=list)

    def update(
        self,
        *,
        signal_price: Optional[float],
        exec_price: Optional[float],
        notional: float,
        ts_routed: Optional[datetime],
        ts_filled: Optional[datetime],
        status: str,
        explicit_slippage: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Update stats from a single order execution.
        """
        self.count += 1

        is_filled = status.upper() in {"FILLED", "OK"} or exec_price is not None
        if is_filled:
            self.filled_count += 1

        # Slippage
        if signal_price and (exec_price is not None):
            if explicit_slippage is not None:
                # explicit_slippage is abs diff between exec and signal
                diff = explicit_slippage
            else:
                diff = exec_price - signal_price

            # slippage in bps (diff / signal_price * 10_000)
            if signal_price > 0:
                bps = (diff / signal_price) * 10_000.0
            else:
                bps = 0.0

            self.slippage_bps.append(float(bps))

            # slippage in currency: diff * qty ~ notional * (diff/signal)
            # but we only have notional = qty*signal → approx:
            abs_slip_val = abs(diff) * (notional / max(signal_price, 1e-8)) if notional > 0 else abs(diff)
            self.slippage_abs.append(float(abs_slip_val))

        # Latency
        if latency_ms is not None:
            self.latency_sec.append(float(latency_ms) / 1000.0)
        elif ts_routed and ts_filled:
            dt = (ts_filled - ts_routed).total_seconds()
            if dt >= 0:
                self.latency_sec.append(float(dt))

    # ---- summary metrics ----

    def avg_slippage_bps(self) -> float:
        return float(statistics.mean(self.slippage_bps)) if self.slippage_bps else 0.0

    def avg_slippage_abs(self) -> float:
        return float(statistics.mean(self.slippage_abs)) if self.slippage_abs else 0.0

    def avg_latency(self) -> float:
        return float(statistics.mean(self.latency_sec)) if self.latency_sec else 0.0

    def fill_ratio(self) -> float:
        return float(self.filled_count) / float(self.count) if self.count > 0 else 0.0

    def score(self) -> float:
        """
        Execution Quality Score heuristic:

            • lower slippage_bps  → better
            • lower latency_sec   → better
            • higher fill_ratio   → better

        Rough range [-1, +1].
        """
        if self.count == 0:
            return 0.0

        sl_bps = self.avg_slippage_bps()
        lat = self.avg_latency()
        fr = self.fill_ratio()

        # slippage: +25 bps → -1, 0 → 0, -25 → +1 (capped)
        sl_component = max(-1.0, min(1.0, -sl_bps / 25.0))

        # latency: 0s → 0, 2s → -0.2, 10s → -1
        lat_component = max(-1.0, min(0.0, -lat / 10.0))

        # fill ratio: 0 → -0.5, 1 → +0.5
        fr_component = fr - 0.5

        total = sl_component + lat_component + fr_component
        return math.tanh(total)


@dataclass
class OrderExecState:
    """
    Reconstructed execution state for a single order_id across multiple events.
    """

    order_id: str
    symbol: Optional[str] = None
    side: Optional[str] = None
    qty: float = 0.0
    signal_price: Optional[float] = None

    broker: Optional[str] = None
    fill_price: Optional[float] = None
    slippage: Optional[float] = None  # absolute diff, if provided
    latency_ms: Optional[float] = None

    ts_submitted: Optional[datetime] = None
    ts_routed: Optional[datetime] = None
    ts_filled: Optional[datetime] = None

    status: str = "UNKNOWN"

    def notional(self) -> float:
        if self.signal_price is not None:
            return self.qty * float(self.signal_price)
        if self.fill_price is not None:
            return self.qty * float(self.fill_price)
        return 0.0


class ExecutionAlphaEngine:
    """
    Phase 80 — Execution Alpha Engine (ledger-aware).

    Workflow:
      1) Call `ingest_event(record)` for each JSON line from order_ledger.jsonl
      2) Call `build_buckets()` to aggregate per (symbol, side, broker)
      3) Use `to_dict()` or `pretty_lines()` for reporting
    """

    def __init__(self) -> None:
        # order_id -> OrderExecState
        self.order_state: Dict[str, OrderExecState] = {}
        # (symbol, side, broker) -> ExecutionBucketStats
        self.buckets: Dict[tuple, ExecutionBucketStats] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _get_state(self, order_id: str) -> OrderExecState:
        if order_id not in self.order_state:
            self.order_state[order_id] = OrderExecState(order_id=order_id)
        return self.order_state[order_id]

    def ingest_event(self, record: Dict[str, Any]) -> None:
        """
        Ingest a single event line from your order_ledger.jsonl.
        """

        rtype = record.get("type")
        if not rtype:
            return

        # Determine order_id
        order_id = record.get("order_id")
        if not order_id and "order" in record:
            order = record["order"]
            order_id = order.get("client_order_id")
        if not order_id:
            return

        state = self._get_state(order_id)

        # Common timestamp
        ts = _parse_ts(record.get("_ts"))

        if rtype == "submitted":
            order = record.get("order", {})
            state.symbol = order.get("symbol") or state.symbol
            state.side = order.get("side") or state.side
            state.qty = float(order.get("qty", state.qty))
            state.signal_price = order.get("signal_price", state.signal_price)
            state.ts_submitted = state.ts_submitted or ts
            state.status = state.status or "SUBMITTED"

        elif rtype == "risk_check":
            order_r = record.get("order_after_risk", {})
            if order_r:
                state.symbol = order_r.get("symbol") or state.symbol
                state.side = order_r.get("side") or state.side
                state.qty = float(order_r.get("qty", state.qty))
                state.signal_price = order_r.get("signal_price", state.signal_price)
            # risk_meta might indicate hard_kill in future; ignore for now

        elif rtype == "reroute":
            route_info = record.get("route_info", {})
            broker = route_info.get("broker")
            if broker:
                state.broker = broker
            route_meta = route_info.get("route_meta", {})
            ts_routed = _parse_ts(route_meta.get("ts_routed"))
            state.ts_routed = state.ts_routed or ts_routed or ts

        elif rtype == "fill":
            fill = record.get("fill", {})
            broker = fill.get("broker")
            if broker:
                state.broker = broker

            raw = fill.get("raw", {})
            fp = fill.get("filled_avg_price") or raw.get("filled_avg_price")
            if fp is not None:
                state.fill_price = float(fp)

            state.ts_filled = state.ts_filled or ts
            state.status = "FILLED"

        elif rtype == "complete":
            summary = record.get("summary", {})
            broker = summary.get("broker")
            if broker:
                state.broker = broker

            fp = summary.get("fill_price")
            if fp is not None:
                state.fill_price = float(fp)

            slip = summary.get("slippage")
            if slip is not None:
                state.slippage = float(slip)

            lat_ms = summary.get("latency_ms")
            if lat_ms is not None:
                state.latency_ms = float(lat_ms)

            route_meta = summary.get("route_meta", {})
            ts_routed = _parse_ts(route_meta.get("ts_routed"))
            if ts_routed and not state.ts_routed:
                state.ts_routed = ts_routed

            state.ts_filled = state.ts_filled or ts
            state.status = "FILLED"

        elif rtype == "error":
            state.status = "ERROR"
            # ctx.order may carry symbol/side/qty
            ctx_order = (record.get("ctx") or {}).get("order") or {}
            if ctx_order:
                state.symbol = ctx_order.get("symbol") or state.symbol
                state.side = ctx_order.get("side") or state.side
                state.qty = float(ctx_order.get("qty", state.qty))
                state.signal_price = ctx_order.get("signal_price", state.signal_price)

        # ignore other types for now

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _get_bucket(self, symbol: str, side: str, broker: str) -> ExecutionBucketStats:
        key = (symbol.upper(), side.upper(), broker.lower() or "unknown")
        if key not in self.buckets:
            self.buckets[key] = ExecutionBucketStats()
        return self.buckets[key]

    def build_buckets(self) -> None:
        """
        Aggregate all order_state into buckets.
        Call this AFTER ingesting all ledger events.
        """
        self.buckets.clear()

        for state in self.order_state.values():
            symbol = (state.symbol or "UNKNOWN").upper()
            side = (state.side or "UNKNOWN").upper()
            broker = (state.broker or "unknown").lower()

            bucket = self._get_bucket(symbol, side, broker)

            bucket.update(
                signal_price=float(state.signal_price) if state.signal_price is not None else None,
                exec_price=float(state.fill_price) if state.fill_price is not None else None,
                notional=state.notional(),
                ts_routed=state.ts_routed,
                ts_filled=state.ts_filled,
                status=state.status,
                explicit_slippage=state.slippage,
                latency_ms=state.latency_ms,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Export all bucket stats as nested dict:
          { symbol: { side: { broker: {...stats...} } } }
        """
        out: Dict[str, Any] = {}
        for (symbol, side, broker), stats in self.buckets.items():
            sym_dict = out.setdefault(symbol, {})
            side_dict = sym_dict.setdefault(side, {})
            side_dict[broker] = {
                "count": stats.count,
                "filled_count": stats.filled_count,
                "fill_ratio": stats.fill_ratio(),
                "avg_slippage_bps": stats.avg_slippage_bps(),
                "avg_slippage_abs": stats.avg_slippage_abs(),
                "avg_latency_sec": stats.avg_latency(),
                "score": stats.score(),
            }
        return out

    def pretty_lines(self) -> List[str]:
        """
        Human-readable table lines for CLI output.
        """
        lines: List[str] = []
        header = (
            "Symbol  Side  Broker    Trades  Filled  Fill%   "
            "Slippage(bps)  Slippage($)  Latency(s)  Score"
        )
        lines.append(header)
        lines.append("-" * len(header))

        data = self.to_dict()
        if not data:
            lines.append("(no execution data yet)")
            return lines

        for symbol, side_map in sorted(data.items()):
            for side, broker_map in sorted(side_map.items()):
                for broker, stats in sorted(broker_map.items()):
                    lines.append(
                        f"{symbol:6s} {side:4s} {broker:8s} "
                        f"{stats['count']:6d} {stats['filled_count']:6d} "
                        f"{stats['fill_ratio']*100:5.1f}% "
                        f"{stats['avg_slippage_bps']:12.2f} "
                        f"{stats['avg_slippage_abs']:11.4f} "
                        f"{stats['avg_latency_sec']:10.3f} "
                        f"{stats['score']:6.3f}"
                    )

        return lines
