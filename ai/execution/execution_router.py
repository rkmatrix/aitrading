from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ai.execution.execution_simulator import ExecutionSimulator, SimParams
from ai.execution.rewards import ExecReward
from ai.execution.execution_policy import ExecutionPolicy, make_obs_vector, ObsSpec


@dataclass
class RouterConfig:
    inventory_limit: float = 500.0
    tick_size: float = 0.01
    # Simulator knobs (mirror phase17.yaml env/simulator merged params)
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
    # Reward knobs (for diagnostics)
    shortfall_weight: float = 1.0
    inventory_weight: float = 5e-4
    latency_weight: float = 1e-3
    fill_bonus: float = 0.05


@dataclass
class SymbolState:
    inventory: float = 0.0
    cash: float = 1_000_000.0
    last_mid: Optional[float] = None
    t: int = 0


class ExecutionRouter:
    """Routes orders to the execution policy and simulates a step."""
    def __init__(self, *, policies: Dict[str, ExecutionPolicy], cfg: RouterConfig,
                 base_cash: float = 1_000_000.0):
        self.policies = policies
        self.cfg = cfg
        self.base_cash = base_cash
        self.state: Dict[str, SymbolState] = {}

        sim_params = SimParams(
            tick_size=cfg.tick_size,
            passive_fill_prob_at_spread=cfg.passive_fill_prob_at_spread,
            improve_fill_boost=cfg.improve_fill_boost,
            join_fill_boost=cfg.join_fill_boost,
            a_bps=cfg.a_bps,
            sigma_bps=cfg.sigma_bps,
            market_slippage_extra_bps=cfg.market_slippage_extra_bps,
            spread_floor_bps=cfg.spread_floor_bps,
            fill_lat_ms_mean=cfg.fill_lat_ms_mean,
            fill_lat_ms_std=cfg.fill_lat_ms_std,
            cancel_prob=cfg.cancel_prob,
        )
        self.sim = ExecutionSimulator(sim_params)
        self.reward = ExecReward(
            shortfall_weight=cfg.shortfall_weight,
            inventory_weight=cfg.inventory_weight,
            latency_weight=cfg.latency_weight,
            fill_bonus=cfg.fill_bonus,
        )
        self.obs_spec = ObsSpec(inventory_limit=cfg.inventory_limit)

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self.state:
            self.state[symbol] = SymbolState(cash=self.base_cash)
        return self.state[symbol]

    def execute(self, order: Dict[str, Any], *, deterministic: bool = False) -> Dict[str, Any]:
        symbol = str(order["symbol"]).upper()
        side = str(order.get("side", "buy")).lower()
        qty = float(order.get("qty", 0.0) or 0.0)
        mid = float(order.get("mid", 0.0) or 0.0)
        spread = float(order.get("spread", 0.01) or 0.01)
        total_steps = int(order.get("total_steps", 2048))

        st = self._get_state(symbol)
        policy = self.policies.get(symbol) or next(iter(self.policies.values()))

        obs = make_obs_vector(
            mid=mid, spread=spread, qty=qty, side=side,
            inventory=st.inventory, t=st.t, total_steps=total_steps,
            last_mid=st.last_mid, spec=self.obs_spec
        )
        action_id = policy.act(obs, deterministic=deterministic)
        side_sign = 1 if side == "buy" else -1

        sim_out = self.sim.step(
            action={0: "market", 1: "join", 2: "improve", 3: "cancel"}[action_id],
            side_sign=side_sign, qty=qty, mid=mid, spread=spread
        )
        filled = bool(sim_out["filled"])
        fill_price = sim_out["fill_price"]
        latency_ms = float(sim_out["latency_ms"]) if sim_out["latency_ms"] is not None else 0.0

        # update balances
        if filled and fill_price is not None:
            st.inventory += side_sign * qty
            st.cash -= side_sign * qty * fill_price
        st.t += 1
        st.last_mid = mid

        r = self.reward(
            side=side_sign,
            mid_before=mid,
            fill_price=fill_price if filled else None,
            inventory=st.inventory,
            latency_ms=latency_ms,
            filled=filled,
        )

        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "mid": mid,
            "spread": spread,
            "action_id": int(action_id),
            "filled": filled,
            "fill_price": fill_price,
            "latency_ms": latency_ms,
            "inventory_after": st.inventory,
            "cash_after": st.cash,
            "reward": float(r),
        }
