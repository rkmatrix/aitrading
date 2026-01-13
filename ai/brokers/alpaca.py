"""
ai/brokers/alpaca.py
-----------------------------------------------------------
Stub + autodetect wrapper for AlpacaBroker.
Detects LIVE / PAPER / DRY_RUN mode and logs a startup banner.
Implements place_order() for PAPER/DRY_RUN simulation when no delegate.
-----------------------------------------------------------
"""
from __future__ import annotations
import os, random, time, logging, inspect, importlib.util
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# try to import a real broker implementation if it exists
try:
    from brokers.broker_alpaca import AlpacaBroker as RealAlpacaBroker
    _REAL = True
except Exception:
    RealAlpacaBroker = None
    _REAL = False


def _alpaca_mode() -> str:
    """detect live / paper / dry_run"""
    if importlib.util.find_spec("alpaca") or importlib.util.find_spec("alpaca_py"):
        return "LIVE" if os.getenv("ALPACA_LIVE_KEY") else "PAPER"
    return "DRY_RUN"


@dataclass
class SimOrder:
    id: str
    symbol: str
    side: str
    qty: int
    order_type: str
    limit_price: Optional[float]
    tag: str
    filled: bool = False
    fill_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class AlpacaBroker:
    def __init__(self, paper: bool = True):
        global _REAL
        self.paper = paper
        self.mode = _alpaca_mode()
        self.delegate = None

        if _REAL:
            try:
                sig = inspect.signature(RealAlpacaBroker.__init__)
                self.delegate = (
                    RealAlpacaBroker(paper=paper)
                    if "paper" in sig.parameters
                    else RealAlpacaBroker()
                )
                logger.info("✅ Using real brokers.broker_alpaca.AlpacaBroker (delegate active)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to init real AlpacaBroker delegate: {e}")
                _REAL = False

        if not self.delegate:
            logger.warning("⚠️ Using STUB AlpacaBroker (paper-simulated trades)")

        # announce mode banner
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"🌐 Broker mode : {self.mode}")
        logger.info(f"🧩 Real module : {'Loaded' if _REAL else 'Stubbed'}")
        logger.info(f"📦 Paper flag  : {self.paper}")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # local simulation state
        self.prices: Dict[str, float] = {}
        self.positions: Dict[str, float] = {}
        self.avg_costs: Dict[str, float] = {}
        self.realized_pnl: Dict[str, float] = {}
        self.orders: Dict[str, SimOrder] = {}

        # sim tuning
        self.latency = 0.05  # 50ms simulated latency

    # -------- price feed simulation --------
    def get_last_price(self, symbol: str) -> float:
        if self.delegate and hasattr(self.delegate, "get_last_price"):
            return self.delegate.get_last_price(symbol)
        base = self.prices.get(symbol, 100.0)
        base *= 1 + random.uniform(-0.002, 0.002)
        base += (100 - base) * 0.001
        self.prices[symbol] = round(base, 2)
        return self.prices[symbol]

    # -------- NEW: place_order (sim) --------
    def place_order(self, order: dict):
        """
        Place a single order. If delegate exists and supports place_order, pass-through.
        Otherwise simulate a fill with light slippage.
        Expected keys: symbol, side, qty, order_type, limit_price, tag
        """
        # delegate pass-through
        if self.delegate and hasattr(self.delegate, "place_order"):
            return self.delegate.place_order(order)

        # simulate locally
        time.sleep(self.latency)
        symbol = order.get("symbol")
        side = str(order.get("side", "")).upper()
        qty = int(order.get("qty", 0))
        order_type = (order.get("order_type") or "MARKET").upper()
        limit_price = order.get("limit_price")

        # choose a reference price
        px_ref = limit_price if (order_type == "LIMIT" and limit_price) else self.get_last_price(symbol)
        slip_factor = random.uniform(-0.05, 0.05) / 100.0  # ±0.05%
        fill_price = round(px_ref * (1 + slip_factor), 4)

        # apply position/pnl
        oid = f"ALP_{int(datetime.utcnow().timestamp()*1000)}"
        sim = SimOrder(
            id=oid, symbol=symbol, side=side, qty=qty,
            order_type=order_type, limit_price=limit_price, tag=order.get("tag", "stub"),
            filled=True, fill_price=fill_price
        )
        self.orders[oid] = sim
        self._apply_fill(sim)
        logger.info(f"🟢 [AlpacaStub] Executed {side} {qty} {symbol} @ {fill_price}")

        return {
            "id": oid,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "fill_price": fill_price,
            "timestamp": datetime.utcnow().isoformat()
        }

    # -------- order routing convenience (legacy) --------
    def route(self, symbol: str, side: str, qty: int,
              order_type: str = "MARKET",
              limit_price: Optional[float] = None,
              tag: str = "stub") -> str:
        if self.delegate and hasattr(self.delegate, "route"):
            return self.delegate.route(symbol=symbol, side=side, qty=qty,
                                       order_type=order_type, limit_price=limit_price, tag=tag)

        oid = f"STUB-{symbol}-{int(time.time()*1000)}"
        px = self.get_last_price(symbol)
        filled = order_type != "LIMIT" or not limit_price or (
            side.upper() == "BUY" and px <= limit_price
        ) or (side.upper() == "SELL" and px >= limit_price)
        fill_price = limit_price if (order_type == "LIMIT" and filled) else px
        self.orders[oid] = SimOrder(oid, symbol, side, qty, order_type, limit_price, tag, filled, fill_price)
        if filled:
            self._apply_fill(self.orders[oid])
            logger.info(f"🧪 [FILLED] {side} {qty}@{fill_price:.2f} {symbol} tag={tag} id={oid}")
        else:
            logger.info(f"⏳ [PENDING] {side} {qty}@{limit_price:.2f} {symbol} tag={tag} id={oid}")
        return oid

    # -------- PnL / position simulation --------
    def _apply_fill(self, order: SimOrder):
        sym = order.symbol
        side = order.side.upper()
        qty = order.qty
        px = order.fill_price
        pos = self.positions.get(sym, 0.0)
        cost = self.avg_costs.get(sym, 0.0)
        rlzd = self.realized_pnl.get(sym, 0.0)

        if side == "BUY":
            new_pos = pos + qty
            new_cost = ((cost * pos) + (px * qty)) / new_pos if new_pos else 0
        else:
            new_pos = pos - qty
            if pos > 0:
                closed = min(qty, pos)
                rlzd += (px - cost) * closed
            new_cost = cost if new_pos > 0 else 0

        self.positions[sym] = new_pos
        self.avg_costs[sym] = new_cost
        self.realized_pnl[sym] = rlzd

    def pretty_pnl_report(self) -> str:
        lines = [f"💰 {self.mode} PnL Summary"]
        total_rlzd = total_unrlzd = 0.0
        for sym, pos in self.positions.items():
            cost = self.avg_costs.get(sym, 0.0)
            px = self.get_last_price(sym)
            unrlzd = (px - cost) * pos
            rlzd = self.realized_pnl.get(sym, 0.0)
            total_rlzd += rlzd
            total_unrlzd += unrlzd
            lines.append(f"• {sym}: qty={pos:.0f}, avg={cost:.2f}, unrlzd={unrlzd:+.2f}, rlzd={rlzd:+.2f}")
        lines.append(f"TOTAL → rlzd={total_rlzd:+.2f}, unrlzd={total_unrlzd:+.2f}")
        return "\n".join(lines)
