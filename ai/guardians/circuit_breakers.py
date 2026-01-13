from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

from ai.state.pnl_tracker import DailyPnlProvider

Side = Literal["BUY", "SELL"]

@dataclass
class RiskPolicy:
    enable_daily_loss_limit: bool = True
    max_daily_loss: float = -250.0
    allow_flatten_only_below_limit: bool = True

@dataclass
class CircuitBreakers:
    policy: RiskPolicy
    pnl_provider: DailyPnlProvider

    def check_new_order_allowed(self, side: Side, is_flattening: bool) -> tuple[bool, Optional[str]]:
        """
        Returns (allowed, reason_if_blocked).
        If daily loss limit is breached:
          - If allow_flatten_only_below_limit: only allow flattening trades (reduce risk/close)
          - Else block all new orders
        """
        if not self.policy.enable_daily_loss_limit:
            return True, None

        pnl = self.pnl_provider.today_pnl()
        if pnl <= self.policy.max_daily_loss:
            if self.policy.allow_flatten_only_below_limit and is_flattening:
                return True, None
            return False, f"Daily loss limit tripped (PnL={pnl:.2f} <= {self.policy.max_daily_loss:.2f})"
        return True, None
