from __future__ import annotations
from typing import Callable, Dict, Any, Optional
from ..policies.base import ExecutionPolicy
from ..policies.rule_based import POV, TWAP
from ..market_features.ob_features import compute_features, LOBSnapshot
from .order_state import OrderStateMachine, ChildOrderParams, MarketContext

class LiveExecutionExecutor:
    """
    Parent-order orchestrator with:
      - policy-driven child sizing/aggression
      - spread/volatility guardrails & auto-fallback
      - cancel/replace child loop via OrderStateMachine
    """
    def __init__(
        self,
        broker,
        policy: ExecutionPolicy,
        *,
        spread_bps_guard: float = 8.0,
        volatility_guard: Optional[float] = 0.06,      # abort/fallback if > 6% intraday vol proxy
        participation_cap: float = 0.15,
        fallback: str = "pov",
        fallback_args: Optional[dict] = None,
        child_max_qty: Optional[int] = None
    ):
        self.broker = broker
        self.policy = policy
        self.spread_bps_guard = spread_bps_guard
        self.volatility_guard = volatility_guard
        self.participation_cap = participation_cap
        self.fallback_name = fallback
        self.fallback_args = fallback_args or {}
        self.child_max_qty = child_max_qty
        self.sm = OrderStateMachine(broker)

        if fallback == "twap":
            self.fallback_factory = lambda total, steps: TWAP(total, steps)
        else:
            pov_part = float(self.fallback_args.get("participation", 0.08))
            pov_min = int(self.fallback_args.get("min_size", 50))
            self.fallback_factory = lambda total, steps: POV(pov_part, pov_min)

    def _choose_price_ref(self, side: str, aggression: str, snap: dict) -> float:
        if aggression == "market":
            return snap["ask"] if side == "buy" else snap["bid"]
        if aggression == "mid":
            return (snap["bid"] + snap["ask"]) / 2.0
        return snap["bid"] if side == "buy" else snap["ask"]

    def execute_parent(self, symbol: str, side: str, total_qty: int, get_lob_snapshot: Callable[[], dict]):
        sent = 0
        steps_left = 200
        fallback_policy: Optional[ExecutionPolicy] = None

        while sent < total_qty and steps_left > 0:
            snap = get_lob_snapshot()
            feats = compute_features(LOBSnapshot(
                bid=snap["bid"], ask=snap["ask"],
                bid_size=snap.get("bid_size", 0), ask_size=snap.get("ask_size", 0),
                volatility=snap.get("volatility", 0.002),
            ))
            state = {
                **feats,
                "remaining_qty": total_qty - sent,
                "steps_left": steps_left,
                "micro_alpha": snap.get("micro_alpha", 0.0),
                "est_volume": snap.get("est_volume", 1000),
            }

            mid = feats["mid"]; spread = feats["spread"]
            spread_bps = 1e4 * spread / max(mid, 1e-6)
            vol = feats["volatility"]

            # guards → fallback
            use_fallback = (spread_bps > self.spread_bps_guard) or (self.volatility_guard and vol > self.volatility_guard)
            pol = self.policy
            if use_fallback:
                if fallback_policy is None:
                    fallback_policy = self.fallback_factory(total_qty, max(steps_left, 1))
                pol = fallback_policy

            action = pol.act(state)
            child_qty = int(min(action.get("size", 0), total_qty - sent))
            if self.child_max_qty:
                child_qty = min(child_qty, self.child_max_qty)
            # participation cap (soft): cap by est volume
            est_vol = max(1, int(state.get("est_volume", 1000)))
            child_qty = int(min(child_qty, self.participation_cap * est_vol))
            if child_qty <= 0:
                steps_left -= 1
                continue

            aggression = action.get("aggression", "mid")
            price_ref = self._choose_price_ref(side, aggression, snap)
            mcx = MarketContext(
                bid=snap["bid"], ask=snap["ask"], spread=spread, mid=mid,
                micro_alpha=state.get("micro_alpha", 0.0)
            )

            # Configure child with modest price nudge (alpha-aware optional hook)
            price_offset_bps = 2.0 if aggression == "mid" else 0.0
            params = ChildOrderParams(
                side=side, qty=child_qty, aggression=aggression,
                max_rest_sec=2.0 if aggression != "market" else 0.5,
                max_age_sec=6.0,
                tif="ioc" if aggression != "passive" else "day",
                price_offset_bps=price_offset_bps,
                max_reprices=2 if aggression != "market" else 0,
                min_partial_fill=1,
                slippage_guard_bps=15.0
            )

            result = self.sm.execute(symbol, params, mcx)
            filled = int(result["filled_qty"])
            sent += filled
            steps_left -= 1

            # Early abort if guards trip or nothing is filling meaningfully
            if result["status"] in ("rejected", "expired", "slip_guard_hit"):
                break

        return {"filled": sent, "requested": total_qty}
