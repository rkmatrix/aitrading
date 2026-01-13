from __future__ import annotations

class Guardian:
    def __init__(self, participation_cap: float = 0.12, max_drawdown_bps: float = 50):
        self.participation_cap = participation_cap
        self.max_drawdown_bps = max_drawdown_bps
        self.dd_bps = 0.0

    def allow(self, state: dict) -> bool:
        if state.get("blocked", False):
            return False
        if state.get("pov", 0.0) > self.participation_cap:
            return False
        if self.dd_bps < -abs(self.max_drawdown_bps):
            return False
        return True

    def update_drawdown(self, pnl_bps_delta: float):
        self.dd_bps = min(self.dd_bps + pnl_bps_delta, 0.0)
