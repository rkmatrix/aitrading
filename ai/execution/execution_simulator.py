from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimParams:
    tick_size: float = 0.01
    passive_fill_prob_at_spread: float = 0.35
    improve_fill_boost: float = 0.4
    join_fill_boost: float = 0.15
    a_bps: float = 3.5
    sigma_bps: float = 2.0
    market_slippage_extra_bps: float = 1.0
    spread_floor_bps: float = 0.2
    fill_lat_ms_mean: float = 120.0
    fill_lat_ms_std: float = 80.0
    cancel_prob: float = 0.08

class ExecutionSimulator:
    """
    Stochastic execution model used by the RL env.
    Actions: 'market', 'join', 'improve', 'cancel'
    """
    def __init__(self, params: SimParams, rng: np.random.Generator | None = None):
        self.p = params
        self.rng = rng or np.random.default_rng(42)

    def _slippage_bps(self, qty: float, market: bool = False) -> float:
        size_norm = max(1e-6, abs(qty))
        base = self.p.a_bps * np.sqrt(size_norm)
        noise = self.rng.normal(0.0, self.p.sigma_bps)
        extra = self.p.market_slippage_extra_bps if market else 0.0
        return float(max(0.0, base + noise + extra))

    def _fill_latency_ms(self) -> float:
        return float(max(0.0, self.rng.normal(self.p.fill_lat_ms_mean, self.p.fill_lat_ms_std)))

    def _passive_fill_prob(self, spread_bps: float, action: str) -> float:
        base = max(self.p.passive_fill_prob_at_spread, self.p.spread_floor_bps)
        boost = 0.0
        if action == "improve":
            boost = self.p.improve_fill_boost
        elif action == "join":
            boost = self.p.join_fill_boost
        p = min(0.95, max(0.01, base + boost) * (1.0 / max(1e-6, spread_bps / self.p.spread_floor_bps)))
        return float(p)

    def step(self, *, action: str, side_sign: int, qty: float, mid: float, spread: float) -> Dict[str, Any]:
        assert action in {"market", "join", "improve", "cancel"}
        spread_bps = (spread / max(1e-9, mid)) * 1e4 if spread and mid else self.p.spread_floor_bps
        tick = self.p.tick_size
        filled = False
        fill_price = None
        latency_ms = 0.0

        if action == "cancel":
            # sometimes cancels fail and we get partial fill
            if self.rng.random() < self.p.cancel_prob:
                action = "join"  # treat as still resting briefly
            else:
                return {"filled": False, "fill_price": None, "latency_ms": 0.0}

        if action == "market":
            # immediate fill at mid +/- half-spread plus slippage
            base_price = mid + side_sign * (spread / 2.0)
            slip_bps = self._slippage_bps(qty, market=True)
            fill_price = base_price + side_sign * (slip_bps / 1e4) * mid
            filled = True
            latency_ms = self._fill_latency_ms() * 0.25  # faster
        else:
            # passive-style; probability of fill depends on spread and improvement
            p_fill = self._passive_fill_prob(spread_bps, action)
            if self.rng.random() < p_fill:
                # got filled with some improvement vs touch
                touch = mid - spread/2.0 if side_sign < 0 else mid + spread/2.0
                improv_ticks = 1 if action == "improve" else 0
                price = touch - side_sign * improv_ticks * tick
                slip_bps = self._slippage_bps(qty, market=False)
                price = price + side_sign * (slip_bps / 1e4) * mid
                fill_price = price
                filled = True
                latency_ms = self._fill_latency_ms()
            else:
                filled = False
                fill_price = None
                latency_ms = self._fill_latency_ms()

        return {"filled": filled, "fill_price": fill_price, "latency_ms": latency_ms}
