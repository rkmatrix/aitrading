# ai/reward/sources.py
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Event:
    ts: float
    symbol: str
    px: float
    position: float      # signed shares
    realized_pnl: float
    unrealized_pnl: float
    slippage: float      # signed currency
    risk: float          # e.g., rolling volatility proxy
    meta: Dict = field(default_factory=dict)

class EventSource:
    def poll(self, max_events: int) -> List[Event]:
        raise NotImplementedError

    def close(self):
        pass


class DummySource(EventSource):
    def __init__(
        self,
        symbols: List[str],
        seed: int = 42,
        start_price: float = 100.0,
        drift_perc: float = 0.0005,
        vol_perc: float = 0.01,
        event_rate_per_symbol: int = 2,
    ):
        self.rng = random.Random(seed)
        self.symbols = symbols
        self.state: Dict[str, Dict] = {}
        for s in symbols:
            self.state[s] = {
                "px": start_price * (0.8 + 0.4 * self.rng.random()),
                "pos": 0.0,
                "cum_realized": 0.0,
                "last_px": None,
                "pnl_unreal": 0.0,
                "risk": 0.01,
            }
        self.drift = drift_perc
        self.vol = vol_perc
        self.rate = max(1, int(event_rate_per_symbol))

    def _step_symbol(self, s: str) -> Event:
        st = self.state[s]
        shock = self.rng.gauss(self.drift, self.vol)
        px = max(0.5, st["px"] * (1.0 + shock))
        trade_shares = self.rng.choice([0, 0, 1, -1, 2, -2])
        old_pos = st["pos"]
        new_pos = old_pos + trade_shares

        realized = 0.0
        if trade_shares != 0 and old_pos != 0 and (old_pos * new_pos) <= 0:
            entry_px = st["last_px"] or st["px"]
            realized = old_pos * (px - entry_px)
            st["cum_realized"] += realized

        st["pos"] = new_pos
        st["last_px"] = px
        st["px"] = px
        st["pnl_unreal"] = new_pos * (px - (st.get("avg_px") or px))
        st["risk"] = max(1e-6, 0.95 * st["risk"] + 0.05 * abs(shock))

        slippage_ccy = px * abs(trade_shares) * (self.rng.uniform(0.0, 0.0003))

        return Event(
            ts=time.time(),
            symbol=s,
            px=px,
            position=new_pos,
            realized_pnl=realized,
            unrealized_pnl=st["pnl_unreal"],
            slippage=slippage_ccy,
            risk=st["risk"],
            meta={"trade_shares": trade_shares},
        )

    def poll(self, max_events: int) -> List[Event]:
        events: List[Event] = []
        if not self.symbols:
            return events
        for s in self.symbols:
            for _ in range(self.rate):
                if len(events) >= max_events:
                    break
                events.append(self._step_symbol(s))
        return events[:max_events]
