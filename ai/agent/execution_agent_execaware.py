"""
ai/agent/execution_agent_execaware.py

Phase 79 — Execution-Aware Live Agent
Phase 86 — Regime-aware volatility metrics
Phase 87 — Micro-pattern features for RL & supervisor
Phase 88.4 — Real execution feedback into MultiPolicySupervisor

- Maintains a rolling (window_size x 5) OHLCV window per symbol.
- Uses MultiPolicySupervisor (Phase 77/84/87/88) + EquityRLPolicyExecAware (Phase 76)
  to decide discrete actions: HOLD=0, BUY=1, SELL=2.
- Sends MARKET orders via SmartOrderRouter v4.
- Feeds real slippage, latency, and per-trade PnL into the supervisor's evolution memory.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from math import floor
from typing import Dict, Any, Deque, List, Tuple

import numpy as np

from ai.data.live_price_router import LivePriceRouter
from ai.policy.multi_policy_supervisor import MultiPolicySupervisor
from ai.execution.smart_order_router import SmartOrderRouter
from ai.features.micro_pattern_detector import compute_micro_features  # Phase 87

logger = logging.getLogger(__name__)


class ExecutionAwareLiveAgent:
    """
    Execution-aware live trading agent that bridges:

        LivePriceRouter → obs window → MultiPolicySupervisor → SmartOrderRouter
    """

    def __init__(
        self,
        symbols: List[str],
        *,
        window_size: int = 60,
        supervisor_cfg: str = "configs/phase77_supervisor.yaml",
        bar_interval: str = "1min",
    ) -> None:
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.window_size = window_size
        self.bar_interval = bar_interval

        self.price_router = LivePriceRouter(bar_interval=bar_interval)
        self.supervisor = MultiPolicySupervisor(supervisor_cfg)
        self.router = SmartOrderRouter()  # uses internal config and env

        # symbol → deque of rows [open, high, low, close, volume]
        self.buffers: Dict[str, Deque[List[float]]] = {
            s: deque(maxlen=window_size) for s in self.symbols
        }

        # basic risk sizing from env
        self.max_notional_per_trade = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "2000"))

        # Phase 88.4: track last fills per symbol to compute per-trade PnL
        #   _last_fill[symbol]      = (side, entry_price, qty) for current open leg
        #   _last_fill_prev[symbol] = previous leg used to compute realized PnL
        self._last_fill: Dict[str, Tuple[str, float, float]] = {}
        self._last_fill_prev: Dict[str, Tuple[str, float, float]] = {}

        logger.info(
            "ExecutionAwareLiveAgent initialized for %d symbols (window=%d, max_notional_per_trade=%.2f)",
            len(self.symbols),
            self.window_size,
            self.max_notional_per_trade,
        )

    # ------------------------------------------------------------------
    # Public loop entry
    # ------------------------------------------------------------------
    def poll_and_step(self) -> None:
        """
        Poll all symbols once, update buffers, possibly route orders.
        Intended to be called from a timed loop (e.g. every POLL_SECONDS).
        """
        for symbol in self.symbols:
            bar = self.price_router.get_latest_bar(symbol)
            if not bar:
                logger.warning("No bar for %s this cycle", symbol)
                continue

            self._handle_bar(symbol, bar)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        # Layout: [open, high, low, close, volume]
        row = [
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            float(bar["volume"]),
        ]
        buf = self.buffers[symbol]
        buf.append(row)

        if len(buf) < self.window_size:
            logger.info(
                "Buffer for %s warming up (%d/%d)",
                symbol,
                len(buf),
                self.window_size,
            )
            return

        # obs shape: (window, 5)
        obs = np.array(buf, dtype=np.float32)

        # ---- Phase 86: compute realized volatility from recent closes ----
        close = obs[:, 3]  # close is 4th column in our layout
        if close.shape[0] > 1:
            rets = np.diff(np.log(close + 1e-8))
            vol = float(np.std(rets)) if rets.size > 0 else 0.0
        else:
            vol = 0.0

        # ---- Phase 87: compute micro-pattern features from obs window ----
        micro_features = compute_micro_features(obs)

        # Simple metrics placeholder (no trade just yet in this step)
        metrics = {
            "slippage": 0.0,
            "latency": 0.0,
            "fill_prob": 1.0,
            "pnl": 0.0,
            "volatility": vol,          # used by MultiPolicySupervisor & regime logic
            "drawdown": 0.0,
            "recent_prices": obs,       # Phase 84/86: regime detector uses this
            "micro_features": micro_features,  # Phase 87: micro-patterns
        }

        out = self.supervisor.choose_action(obs, metrics)
        action_raw = out["action"]
        chosen_policy = out.get("chosen_policy")

        if isinstance(action_raw, np.ndarray):
            if action_raw.shape == () or action_raw.shape == ():
                action = int(action_raw)
            elif action_raw.shape[0] == 1:
                action = int(action_raw[0])
            else:
                action = int(np.argmax(action_raw))
        else:
            action = int(action_raw)

        logger.info(
            "ExecAwareAgent: %s policy=%s decided action=%d @ price=%.4f (vol=%.6f, vol_ratio=%.6f, vol_spike=%.3f)",
            symbol,
            chosen_policy,
            action,
            bar["close"],
            vol,
            micro_features.get("vol_ratio_20_60", 1.0),
            micro_features.get("volume_ratio", 1.0),
        )

        self._execute_action(symbol, action, float(bar["close"]))

    def _compute_trade_pnl(self, side: str, entry: float, exit: float, qty: float) -> float:
        """
        Phase 88.4:
        Computes per-trade PnL for evolution learning.
        Simple: (exit - entry) * qty for BUY,
                (entry - exit) * qty for SELL.
        """
        if qty <= 0:
            return 0.0
        if entry <= 0 or exit <= 0:
            return 0.0
        if side.upper() == "BUY":
            return (exit - entry) * qty
        else:
            return (entry - exit) * qty

    def _execute_action(self, symbol: str, action: int, price: float) -> None:
        """
        Map discrete action → order via SmartOrderRouter.

        0 = HOLD → no order
        1 = BUY  → MARKET buy with size derived from max_notional_per_trade
        2 = SELL → MARKET sell with same sizing rule (may rely on broker/risk guard)
        """
        if action == 0:
            return

        qty = self._compute_position_size(price)
        if qty <= 0:
            logger.warning(
                "ExecAwareAgent: Computed qty <= 0 for %s at price %.4f; skipping",
                symbol,
                price,
            )
            return

        side = "BUY" if action == 1 else "SELL"
        tag = "phase79_execaware"

        logger.info(
            "ExecAwareAgent: Routing %s %s %.2f @ %.4f",
            side,
            symbol,
            qty,
            price,
        )

        # Extra info for router (can be used by prediction models / journal)
        extra = {
            "signal_price": price,
        }

        res = self.router.route_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="MARKET",
            tag=tag,
            extra=extra,
        )

        if res.get("status") == "OK":
            # Phase 88.4: Forward real execution metrics into evolution memory
            fill = res.get("fill") or {}
            exec_metrics = res.get("execution_metrics") or {}

            filled_price = float(fill.get("filled_avg_price", 0.0))

            metrics = {
                "slippage": float(exec_metrics.get("slippage", 0.0)),
                "latency": float(exec_metrics.get("latency_ms", 0.0)),
                "fill_prob": 1.0,  # assume 1 for market orders
                "pnl": 0.0,        # will be updated from prior leg if available
            }

            # Store entry price for PnL on the NEXT trade
            self._last_fill[symbol] = (side, filled_price, qty)

            # If we had a previous leg, compute realized PnL between legs
            if symbol in self._last_fill_prev:
                prev_side, prev_price, prev_qty = self._last_fill_prev[symbol]
                pnl = self._compute_trade_pnl(prev_side, prev_price, filled_price, prev_qty)
                metrics["pnl"] = pnl

            # Move current leg into "previous" slot for next time
            self._last_fill_prev[symbol] = self._last_fill[symbol]

            # Feed into supervisor evolution loop (uses Option C + Phase 88.3 memory)
            # Note: this calls a "protected" method by design; it's purposeful so
            #       we can update performance with real trade metrics.
            try:
                self.supervisor._update_perf_from_metrics(metrics)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("ExecAwareAgent: failed to update supervisor perf metrics: %s", e)

        else:
            logger.error("ExecAwareAgent: Router error for %s → %s", symbol, res)

    def _compute_position_size(self, price: float) -> float:
        """
        Basic risk-aware sizing based on MAX_NOTIONAL_PER_TRADE.
        """
        if price <= 0:
            return 0.0
        shares = floor(self.max_notional_per_trade / price)
        return float(shares)
