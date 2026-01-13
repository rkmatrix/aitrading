import asyncio
import math
import logging
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any

from tools.telegram_alerts import send_order_alert, send_pnl_alert
from tools.env_utils import current_mode_from_env
from ai.execution.broker_alpaca_live import AlpacaLiveBroker
from ai.marketdata.tick_stream import CSVTickStream, AlpacaTickStream, Tick
from ai.execution.trade_ledger import TradeLedger
from ai.execution.pnl_bridge_live import log_fill

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses and Managers
# --------------------------------------------------------------------------- #
@dataclass
class OrderIntent:
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    type: str          # "market" | "limit"
    limit_price: Optional[float] = None
    reference_mid: Optional[float] = None


class RiskManager:
    """Simple guardrails before routing any order."""
    def __init__(self, cfg: Dict[str, Any]):
        r = cfg.get("risk", {})
        self.max_dollars_per_order = float(r.get("max_dollars_per_order", 5000))
        self.max_qty_per_order = int(r.get("max_qty_per_order", 50))
        self.max_open_orders_per_symbol = int(r.get("max_open_orders_per_symbol", 3))
        self.max_notional_per_symbol = float(r.get("max_notional_per_symbol", 20000))
        self.open_orders: Dict[str, int] = {}

    def allow(self, symbol: str, qty: float, price: float) -> bool:
        notional = abs(qty * price)
        if notional > self.max_dollars_per_order:
            return False
        if abs(qty) > self.max_qty_per_order:
            return False
        if self.open_orders.get(symbol, 0) >= self.max_open_orders_per_symbol:
            return False
        return True

    def mark_open(self, symbol: str):
        self.open_orders[symbol] = self.open_orders.get(symbol, 0) + 1

    def mark_closed(self, symbol: str):
        self.open_orders[symbol] = max(0, self.open_orders.get(symbol, 0) - 1)


# --------------------------------------------------------------------------- #
# Real-Time Execution Adapter
# --------------------------------------------------------------------------- #
class RealTimeExecutionAdapter:
    def __init__(self, cfg: Dict[str, Any], secrets: Dict[str, str]):
        self.cfg = cfg
        self.secrets = secrets
        self.mode = current_mode_from_env(cfg.get("mode", "PAPER"))
        self.symbols = cfg["symbols"]
        self.order_defaults = cfg.get("order_defaults", {})
        self.alerts_cfg = cfg.get("alerts", {})
        self.risk = RiskManager(cfg)
        self.ledger = TradeLedger()   # <-- in-memory ledger
        self._broker = None
        self._stream = None

    # ------------------------------------------------------------------- #
    def _mk_broker(self):
        if self.mode == "DRY_RUN":
            return None
        api_key = self.secrets.get("ALPACA_API_KEY", "")
        api_secret = self.secrets.get("ALPACA_API_SECRET", "")
        paper = (self.mode == "PAPER")
        return AlpacaBroker(api_key, api_secret, paper=paper)

    def _mk_stream(self):
        streams = self.cfg.get("streams", {})
        provider = streams.get("provider", "ALPACA").upper()
        if self.mode == "DRY_RUN" or provider == "CSV":
            return CSVTickStream(self.cfg["streams"]["csv_glob"], self.symbols)
        api_key = self.secrets.get("ALPACA_API_KEY", "")
        api_secret = self.secrets.get("ALPACA_API_SECRET", "")
        return AlpacaTickStream(api_key, api_secret, self.symbols)

    # ------------------------------------------------------------------- #
    async def run(self, intents_queue: "asyncio.Queue[OrderIntent]"):
        """Run tick and intent consumers concurrently."""
        self._broker = self._mk_broker()
        self._stream = self._mk_stream()

        async def tick_task():
            async for tick in self._stream.stream():
                await self.on_tick(tick)

        async def intent_task():
            while True:
                intent = await intents_queue.get()
                await self.on_intent(intent)

        await asyncio.gather(tick_task(), intent_task())

    async def on_tick(self, tick: Tick):
        # Currently no internal signal generation;
        # external agents push OrderIntents to the queue.
        pass

    # ------------------------------------------------------------------- #
    async def on_intent(self, intent: OrderIntent):
        """Handle incoming trade intent from strategy layer."""
        typ = (intent.type or self.order_defaults.get("type", "limit")).lower()
        tif = self.order_defaults.get("time_in_force", "day").lower()

        # Derive limit price if missing
        limit_px = intent.limit_price
        if typ == "limit" and (limit_px is None):
            bps = float(self.order_defaults.get("limit_slippage_bps", 5))
            ref_mid = intent.reference_mid or 0
            adj = ref_mid * (bps / 10_000.0)
            limit_px = round(ref_mid + (adj if intent.side.lower() == "buy" else -adj), 2)

        qty_round = int(self.order_defaults.get("qty_round", 1))
        qty = math.copysign(max(1, round(abs(intent.qty) / qty_round) * qty_round), intent.qty)

        price_for_risk = limit_px or intent.reference_mid or 0
        if not self.risk.allow(intent.symbol, qty, abs(price_for_risk)):
            send_order_alert("ERROR", {"reason": "Risk blocked", "symbol": intent.symbol, "qty": qty})
            return

        # ------------------- DRY RUN -------------------
        if self.mode == "DRY_RUN":
            fill_price = float(limit_px or intent.reference_mid or 0)
            ts = dt.datetime.now()

            send_order_alert("ORDER_SUBMITTED", {
                "symbol": intent.symbol, "side": intent.side,
                "qty": qty, "price": fill_price, "type": typ, "mode": self.mode
            })
            send_order_alert("ORDER_FILLED", {
                "symbol": intent.symbol, "side": intent.side,
                "qty": qty, "price": fill_price, "order_id": f"DRY_{intent.symbol}"
            })

            # Update ledger and persist if position closes
            closed = self.ledger.apply_fill(
                symbol=intent.symbol,
                side=intent.side,
                qty=abs(qty),
                price=fill_price,
                ts=ts,
            )
            if closed:
                send_pnl_alert(
                    closed["symbol"],
                    closed["pnl"],
                    closed["entry_price"],
                    closed["exit_price"],
                    opened_at=closed["opened_at"],
                )
                # Persist to CSV + weekly aggregate
                log_fill(
                    closed["symbol"],
                    closed["pnl"],
                    closed["entry_price"],
                    closed["exit_price"],
                    closed["opened_at"],
                    closed["closed_at"],
                    closed["duration_seconds"],
                )
            return

        # ------------------- LIVE/PAPER -------------------
        try:
            self.risk.mark_open(intent.symbol)
            res: OrderResult = self._broker.submit_order(
                symbol=intent.symbol,
                side=intent.side,
                qty=abs(qty),
                order_type=typ,
                tif=tif,
                limit_price=limit_px,
            )

            send_order_alert("ORDER_SUBMITTED", {
                "symbol": res.symbol,
                "side": res.side,
                "qty": res.qty,
                "type": res.type,
                "limit_price": res.limit_price,
                "order_id": res.id,
                "status": res.status,
            })

            # Poll for fill
            for _ in range(60):
                await asyncio.sleep(1)
                r2 = self._broker.get_order(res.id)
                if r2.status in ("filled", "partially_filled", "canceled", "expired"):
                    ev = "ORDER_FILLED" if r2.status in ("filled", "partially_filled") else "ORDER_CANCELED"
                    send_order_alert(ev, {
                        "symbol": r2.symbol,
                        "side": r2.side,
                        "qty": r2.qty,
                        "filled_qty": r2.filled_qty,
                        "status": r2.status,
                        "limit_price": r2.limit_price,
                        "order_id": r2.id,
                    })

                    # Use limit_price as a fallback for fill price
                    fill_price = float(r2.limit_price or 0.0)
                    ts = dt.datetime.now()

                    # Update ledger and emit PnL mini-alert if closed
                    closed = self.ledger.apply_fill(
                        symbol=r2.symbol,
                        side=r2.side,
                        qty=abs(r2.filled_qty or r2.qty),
                        price=fill_price,
                        ts=ts,
                    )
                    if closed:
                        send_pnl_alert(
                            closed["symbol"],
                            closed["pnl"],
                            closed["entry_price"],
                            closed["exit_price"],
                            opened_at=closed["opened_at"],
                        )
                        log_fill(
                            closed["symbol"],
                            closed["pnl"],
                            closed["entry_price"],
                            closed["exit_price"],
                            closed["opened_at"],
                            closed["closed_at"],
                            closed["duration_seconds"],
                        )
                    break

        except Exception as e:
            log.exception("Order error")
            send_order_alert("ERROR", {"reason": str(e), "symbol": intent.symbol})
        finally:
            self.risk.mark_closed(intent.symbol)
