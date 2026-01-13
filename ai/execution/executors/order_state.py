from __future__ import annotations
import time, math, sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from ai.execution.utils.order_logger import log_order


@dataclass
class ChildOrderParams:
    side: str                # 'buy' | 'sell'
    qty: int
    aggression: str          # 'passive' | 'mid' | 'market'
    max_rest_sec: float = 2.0
    max_age_sec: float = 10.0
    tif: str = "ioc"
    price_offset_bps: float = 0.0
    max_reprices: int = 3
    min_partial_fill: int = 1
    slippage_guard_bps: Optional[float] = 10.0
    max_total_wait_sec: float = 15.0     # 💡 total wait timeout across all retries


@dataclass
class MarketContext:
    bid: float
    ask: float
    spread: float
    mid: float
    micro_alpha: float = 0.0


class OrderStateMachine:
    """
    Executes one child order with cancel/replace logic.
    - Logs progress to console
    - Times out gracefully if Alpaca hangs/retries
    - Records all events to order log
    """

    def __init__(self, broker, *, poll_interval: float = 0.3):
        self.broker = broker
        self.poll_interval = poll_interval

    # ------------------------------------------------------------------
    def _limit_from_aggression(self, params: ChildOrderParams, mcx: MarketContext) -> tuple[str, Optional[float], str]:
        """Compute initial order type, price, and TIF."""
        if params.aggression == "market":
            return "market", None, "ioc"
        if params.aggression == "mid":
            px = mcx.mid + (params.price_offset_bps / 1e4) * mcx.mid
        else:
            if params.side == "buy":
                px = mcx.bid + (params.price_offset_bps / 1e4) * mcx.bid
            else:
                px = mcx.ask - (params.price_offset_bps / 1e4) * mcx.ask
        # ensure legal tick increment
        px = round(px, 2)
        tif = "ioc" if params.aggression != "passive" else "day"
        return "limit", px, tif

    # ------------------------------------------------------------------
    def _violates_slip_guard(self, filled_avg_price: float, ref_px: float, side: str, guard_bps: float) -> bool:
        if guard_bps is None:
            return False
        slip = (filled_avg_price - ref_px) / ref_px
        slip_bps = 1e4 * (slip if side == "buy" else -slip)
        return slip_bps > guard_bps

    # ------------------------------------------------------------------
    def execute(self, symbol: str, params: ChildOrderParams, mcx: MarketContext) -> Dict[str, Any]:
        """Run full cancel/replace lifecycle for one child order."""
        order_type, price, tif = self._limit_from_aggression(params, mcx)
        ref_px = mcx.ask if params.side == "buy" else mcx.bid
        if params.aggression == "mid":
            ref_px = mcx.mid

        start_ts = time.time()
        reprices = 0
        total_wait = 0.0

        # --- place initial order ---
        print(f"🟢 Placing {params.side.upper()} {params.qty} {symbol} ({order_type}/{tif}) @ {price or 'MKT'}")
        try:
            last = self.broker.place(symbol, params.side, params.qty, order_type, price, tif=tif)
            oid = last.get("order_id")
            log_order("place_init", last)
        except Exception as e:
            print(f"❌ Failed to place order: {e}")
            log_order("place_failed", {"symbol": symbol, "side": params.side, "error": str(e)})
            return {"filled_qty": 0, "avg_price": 0.0, "status": "error", "error": str(e)}

        filled_qty = 0
        filled_value = 0.0
        placed_ts = time.time()

        # --- monitoring loop ---
        while True:
            time.sleep(self.poll_interval)
            total_wait = time.time() - start_ts

            # Global guard — if too long overall, abort
            if total_wait > params.max_total_wait_sec:
                print(f"⏰ Timeout ({total_wait:.1f}s) reached for {symbol}, canceling...")
                try:
                    self.broker.cancel(oid)
                    log_order("timeout_cancel", {"order_id": oid, "symbol": symbol})
                except Exception:
                    pass
                return {"filled_qty": filled_qty, "avg_price": 0.0, "status": "timeout"}

            try:
                print(f"🔄 Checking order {oid} ...", end="\r", flush=True)
                st = self.broker.get_status(oid)
            except Exception as e:
                print(f"\n⚠️  Error during get_status: {e}")
                time.sleep(1)
                continue

            status = (st.get("status") or "").lower()
            fqty = int(st.get("filled_qty", 0) or 0)
            favg = float(st.get("filled_avg_price", 0) or 0)

            if fqty > 0:
                filled_qty = fqty
                filled_value = favg * fqty
                log_order("partial_fill", st)
                print(f"💰 Partial fill: {filled_qty} @ {favg:.2f}")

            # terminal conditions
            if status in ("filled", "canceled", "rejected", "expired"):
                print(f"✅ Final status: {status.upper()} for {symbol} (qty={filled_qty})")
                log_order("final_status", st)
                break

            # time since placement
            age = time.time() - placed_ts
            if age >= params.max_age_sec:
                print(f"⚠️  Max age reached ({age:.1f}s) → canceling {oid}")
                self.broker.cancel(oid)
                log_order("max_age_cancel", {"order_id": oid, "symbol": symbol})
                status = "canceled"
                break

            # Replace logic
            if age >= params.max_rest_sec and reprices < params.max_reprices and order_type == "limit":
                nudge = round(0.5 * mcx.spread, 2)
                new_px = round((price or mcx.bid) + (nudge if params.side == "buy" else -nudge), 2)
                print(f"🔁 Replacing {oid} → new price {new_px}")
                try:
                    rep = self.broker.replace(oid, limit_price=new_px)
                    log_order("replace_attempt", rep)
                    oid = rep.get("order_id") or oid
                    price = new_px
                    placed_ts = time.time()
                    reprices += 1
                except Exception as e:
                    print(f"⚠️  Replace failed: {e}")
                continue

        # --- finalize ---
        avg_price = (filled_value / filled_qty) if filled_qty > 0 else 0.0
        if filled_qty > 0 and params.slippage_guard_bps is not None:
            if self._violates_slip_guard(avg_price, ref_px, params.side, params.slippage_guard_bps):
                print(f"🚨 Slippage guard triggered ({avg_price:.2f} vs ref {ref_px:.2f})")
                log_order("slip_guard_hit", {"order_id": oid, "symbol": symbol, "avg_price": avg_price})
                return {"filled_qty": filled_qty, "avg_price": avg_price, "status": "slip_guard_hit"}

        return {"filled_qty": filled_qty, "avg_price": avg_price, "status": status or "unknown", "order_id": oid}
