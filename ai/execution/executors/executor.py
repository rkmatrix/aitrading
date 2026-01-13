from __future__ import annotations
from typing import Dict, Any, Optional, Callable
from ..policies.base import ExecutionPolicy
from ..market_features.ob_features import compute_features, LOBSnapshot
from ..policies.rule_based import TWAP, POV

class ExecutionExecutor:
    """
    Orchestrates child-order decisions with safety rails.
    Auto-fallback to TWAP/POV if conditions breach thresholds.
    """
    def __init__(
        self,
        broker,
        policy: ExecutionPolicy,
        spread_bps_guard: float = 8.0,      # if spread > 8 bps of mid, get conservative
        min_fill_step: int = 1,
        fallback: str = "pov",              # "twap" or "pov"
        fallback_args: Optional[dict] = None,
    ):
        self.broker = broker
        self.policy = policy
        self.spread_bps_guard = spread_bps_guard
        self.min_fill_step = min_fill_step
        self.fallback_name = fallback
        self.fallback_args = fallback_args or {}

        if fallback == "twap":
            self._fallback_factory: Callable[[int,int], ExecutionPolicy] = lambda total, steps: TWAP(total, steps)
        else:
            pov_part = float(self.fallback_args.get("participation", 0.08))
            pov_min = int(self.fallback_args.get("min_size", 50))
            self._fallback_factory = lambda total, steps: POV(pov_part, pov_min)

    def _choose_order_type(self, side: str, agg: str, snap: dict) -> tuple[str, Optional[float]]:
        if agg == "market":
            return "market", None
        if agg == "mid":
            price = (snap["bid"] + snap["ask"]) / 2.0
            return "limit", price
        # passive
        price = snap["bid"] if side == "buy" else snap["ask"]
        return "limit", price

    def execute_parent(self, symbol: str, side: str, total_qty: int, get_lob_snapshot: Callable[[], dict]):
        sent = 0
        steps_left = 100
        fallback_policy: Optional[ExecutionPolicy] = None

        while sent < total_qty and steps_left > 0:
            snap = get_lob_snapshot()
            feats = compute_features(LOBSnapshot(**{
                "bid": snap["bid"], "ask": snap["ask"],
                "bid_size": snap.get("bid_size", 0),
                "ask_size": snap.get("ask_size", 0),
                "volatility": snap.get("volatility", 0.002),
            }))
            state = {**feats,
                     "remaining_qty": total_qty - sent,
                     "steps_left": steps_left,
                     "micro_alpha": snap.get("micro_alpha", 0.0),
                     "est_volume": snap.get("est_volume", 1000)}

            # guard-rail: spread in bps
            mid = feats["mid"]
            spread_bps = 1e4 * feats["spread"] / max(mid, 1e-6)

            use_fallback = spread_bps > self.spread_bps_guard
            pol = self.policy
            if use_fallback:
                if fallback_policy is None:
                    fallback_policy = self._fallback_factory(total_qty, max(steps_left, 1))
                pol = fallback_policy

            action = pol.act(state)
            size = int(min(max(action.get("size", 0), self.min_fill_step), total_qty - sent))
            if size <= 0:
                steps_left -= 1
                continue

            order_type, px = self._choose_order_type(side, action.get("aggression", "mid"), snap)
            r = self.broker.place(symbol, side, size, order_type, px)
            # In a real integration, you would poll order status, handle partials/cancel/replace, etc.
            filled = int(r.get("filled_qty", size)) if r.get("status") == "filled" else 0
            sent += filled
            steps_left -= 1

        return {"filled": sent, "requested": total_qty}
