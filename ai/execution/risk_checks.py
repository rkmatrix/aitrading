from typing import Dict, Tuple
from ai.execution.order_types import OrderRequest
from ai.utils.log_utils import get_logger

class RiskChecks:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger("RiskChecks")

    def check_spread(self, quote: dict) -> Tuple[bool, str]:
        max_spread = float(self.cfg["pre_trade"]["max_spread_bps"])
        bid, ask = quote["bid"], quote["ask"]
        if bid <= 0 or ask <= 0 or ask <= bid:
            return False, "bad-quote"
        spread_bps = (ask - bid) / ((ask + bid)/2) * 10000
        return (spread_bps <= max_spread, f"spread_bps={spread_bps:.2f}")

    def check_slippage(self, side: str, quote: dict, limit_px: float | None) -> Tuple[bool, str]:
        max_slip = float(self.cfg["pre_trade"]["max_slippage_bps"])
        ref = quote["last"]
        tgt = (limit_px if limit_px is not None else (quote["ask"] if side=="BUY" else quote["bid"]))
        slip_bps = abs(tgt - ref) / ref * 10000
        return (slip_bps <= max_slip, f"slip_bps={slip_bps:.2f}")

    def check_order_limits(self, req: OrderRequest, px: float, acct_cfg: Dict) -> Tuple[bool, str]:
        notional = req.qty * px
        if notional > float(acct_cfg["max_order_notional"]):
            return False, "order_notional_exceeds_limit"
        return True, "ok"

    def run_all(self, req: OrderRequest, quote: dict, acct_cfg: Dict) -> Tuple[bool, str]:
        ok1, r1 = self.check_spread(quote)
        if not ok1: return False, f"spread_violation:{r1}"
        ok2, r2 = self.check_slippage(req.side, quote, req.limit_price)
        if not ok2: return False, f"slippage_violation:{r2}"
        ok3, r3 = self.check_order_limits(req, quote["last"], acct_cfg)
        if not ok3: return False, f"limits_violation:{r3}"
        return True, "ok"
