from __future__ import annotations
import time
from typing import Dict, Optional
from ai.utils.log_utils import get_logger
from ai.execution.exec_styles import VWAPStyle, POVStyle, SmartSliceStyle
from ai.execution.exec_styles.base import MarketState
from ai.execution.child_order_book import ChildOrderBook
from ai.execution.order_types import OrderRequest, OrderResult
from ai.execution.order_router import SmartOrderRouter
from ai.execution.risk_checks import RiskChecks
from ai.execution.analytics import choose_style

LOG = get_logger("AdaptiveExecutor")

class AdaptiveExecutionEngine:
    """
    Given a parent order (symbol/side/qty), emits child orders over time using
    a selected style (VWAP/POV/SMART_SLICE), routes via SmartOrderRouter, and
    aggregates fills until done.
    """
    def __init__(self, cfg: Dict, router: SmartOrderRouter, risk: RiskChecks):
        self.cfg = cfg
        self.router = router
        self.risk = risk

    def _build_style(self, name: str, params: Dict, baseline_vol_per_sec: float) -> object:
        p = dict(self.cfg["execution"]["style_params"].get(name, {}))
        p.update(params or {})
        if name == "VWAP":
            return VWAPStyle(p)
        if name == "POV":
            return POVStyle(p, baseline_vol_per_sec=baseline_vol_per_sec)
        if name == "SMART_SLICE":
            return SmartSliceStyle(p)
        raise ValueError(f"Unknown style {name}")

    def execute(self, symbol: str, side: str, qty: int, signal_strength: float | None,
                adapters: Dict[str, object]) -> Dict:
        remaining = int(qty)
        book = ChildOrderBook()
        default_style = self.cfg["execution"].get("default_style", "SMART_SLICE")
        baseline = float(self.cfg["analytics"].get("baseline_vol_per_sec", 2500.0))

        # Estimate simple liquidity from candidate venue
        vname = self.router.select_venue(symbol) or self.cfg["venues"]["failover"]
        quote = adapters[vname].get_quote(symbol)
        est_liq = min(float(quote.get("size", 100)) / 1000.0, 1.0)

        decision = choose_style(default_style, signal_strength, est_liq)
        style = self._build_style(decision.style, decision.params, baseline)
        LOG.info("Exec style=%s params=%s", decision.style, decision.params)

        started = time.time()
        while remaining > 0:
            vname = self.router.select_venue(symbol) or self.cfg["venues"]["failover"]
            quote = adapters[vname].get_quote(symbol)
            state = MarketState(symbol=symbol, side=side, remaining_qty=remaining, quote=quote, now_ts=time.time())
            children = style.next_children(state)
            for child in children:
                ok, why = self.risk.run_all(child, quote, self.cfg["account"])
                if not ok:
                    LOG.debug("Child risk reject: %s", why)
                    continue
                res: OrderResult = self.router.route(child)
                if res and res.ok:
                    book.add_fill(res)
                    remaining = max(0, remaining - res.filled_qty)
                # else rejected or partials handled by venue adapter
            if style.done(state):
                break
            time.sleep(max(0.2, 1.0 / int(self.cfg.get("clock_hz", 2))))

        return {
            "filled_qty": book.filled_qty,
            "filled_vwap": book.vwap,
            "elapsed_sec": time.time() - started,
            "style": decision.style
        }
