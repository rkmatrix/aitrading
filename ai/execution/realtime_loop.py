import asyncio
import logging
import datetime as dt
import signal
import inspect
from typing import Dict, Any, List, Optional

from ai.allocators.portfolio_brain import PortfolioBrain
from ai.execution.smart_order_router import SmartOrderRouter
from ai.guardian.risk_guardian import RiskGuardian
from ai.execution.data_feed_live import DataFeedLive
from ai.execution.order_state_tracker import OrderStateTracker
from ai.execution.telemetry_bridge import TelemetryBridge
from tools.telegram_alerts import send_trade_alert, send_pnl_alert, send_system_alert

logger = logging.getLogger("RealTimeExecutionLoop")


class RealTimeExecutionLoop:
    """
    Phase 25.2 — Real-Time Execution Loop with:
      • Alpaca IEX WebSocket + yfinance fallback
      • Circuit-breaker and auto-cooldown / recovery
      • RiskGuardian enforcement + Telegram alerts
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = cfg.get("mode", "DRY_RUN")
        self.poll = int(cfg.get("poll_interval", 5))
        self.symbols: List[str] = list(cfg.get("symbols", []))

        # --- subsystems ---
        self.brain = PortfolioBrain(cfg)
        self.router = self._init_router(cfg)
        self.guardian = RiskGuardian(cfg)
        self.tracker = OrderStateTracker()
        self.telemetry = TelemetryBridge(cfg)
        self.feed = DataFeedLive(cfg)

        # --- breaker / cooldown parameters ---
        br = cfg.get("breaker", {})
        self.error_limit = int(br.get("error_limit", 5))
        self.hard_drawdown_mult = float(br.get("hard_dd_mult", 2.0))
        self.hard_drawdown = (
            float(cfg.get("risk_guardian", {}).get("max_drawdown", 0.05))
            * self.hard_drawdown_mult
        )

        cd = cfg.get("cooldown", {})
        self.cooldown_enabled = bool(cd.get("enabled", True))
        self.cooldown_minutes = int(cd.get("minutes", 15))
        self.cooldown_on_error = bool(cd.get("on_error_breaker", True))
        self.cooldown_on_dd = bool(cd.get("on_drawdown_breaker", True))

        self.state = "RUNNING"
        self.cooldown_until: Optional[dt.datetime] = None
        self.error_count = 0
        self.stop_signal = False

        signal.signal(signal.SIGINT, self._graceful_stop)
        signal.signal(signal.SIGTERM, self._graceful_stop)

        logger.info(
            f"🚀 RealTimeExecutionLoop initialized (mode={self.mode}) symbols={self.symbols}"
        )

    # ---------- setup ----------
    def _graceful_stop(self, *_):
        logger.warning("🛑 Stop signal received, shutting down gracefully…")
        self.stop_signal = True

    def _init_router(self, cfg: Dict[str, Any]):
        sig = inspect.signature(SmartOrderRouter.__init__)
        params = sig.parameters
        if "brokers" in params and "config" in params:
            return SmartOrderRouter(brokers=["alpaca"], config=cfg)
        if "config" in params:
            return SmartOrderRouter(config=cfg)
        if "cfg" in params:
            return SmartOrderRouter(cfg)
        return SmartOrderRouter(cfg)

    async def _route_order_compat(self, sym: str, sig: Dict[str, Any]) -> Dict[str, Any]:
        for method in ("route", "submit", "send"):
            if hasattr(self.router, method):
                return getattr(self.router, method)(sym, sig)
        raise AttributeError("SmartOrderRouter missing route/submit/send methods.")

    # ---------- cooldown helpers ----------
    def _enter_cooldown(self, reason: str):
        if not self.cooldown_enabled:
            logger.warning(f"⚠️ Breaker hit ({reason}) but cooldown disabled — stopping loop.")
            self.stop_signal = True
            return
        self.state = "COOLDOWN"
        self.cooldown_until = dt.datetime.utcnow() + dt.timedelta(minutes=self.cooldown_minutes)
        logger.error(f"🧊 Entering COOLDOWN for {self.cooldown_minutes} min — reason: {reason}")
        asyncio.create_task(
            send_system_alert(f"Entering cooldown ({reason}) for {self.cooldown_minutes} minutes.")
        )

    def _maybe_exit_cooldown(self):
        if self.state == "COOLDOWN" and self.cooldown_until and dt.datetime.utcnow() >= self.cooldown_until:
            self.state = "RUNNING"
            self.error_count = 0
            logger.info("🌤️ Exiting COOLDOWN — resuming trading.")
            asyncio.create_task(send_system_alert("Exiting cooldown — resuming trading."))

    # ---------- main loop ----------
    async def start(self):
        logger.info("✅ Starting real-time loop…")
        await self.feed.connect(self.symbols)
        while not self.stop_signal:
            try:
                if self.state == "COOLDOWN":
                    self._maybe_exit_cooldown()
                    await asyncio.sleep(self.poll)
                    continue

                await self.step()
                self.error_count = 0
            except Exception as e:
                self.error_count += 1
                logger.exception(f"💥 Loop error #{self.error_count}: {e}")
                if self.error_count >= self.error_limit and self.cooldown_on_error:
                    self._enter_cooldown("error_breaker")
            await asyncio.sleep(self.poll)

        logger.info("🧩 Real-time loop stopped cleanly.")

    # ---------- single tick ----------
    async def step(self):
        ts = dt.datetime.utcnow().isoformat()
        logger.info(f"🕒 Tick @ {ts}")

        # 1️⃣ Prices
        if hasattr(self.brain, "fetch_prices"):
            prices: Dict[str, float] = await self.brain.fetch_prices(self.symbols)
        else:
            prices: Dict[str, float] = await self.feed.fetch_prices(self.symbols)
        if not prices:
            logger.warning("⚠️ No live prices available, skipping tick.")
            return
        self.telemetry.emit("prices_fetched", 1, symbols=len(prices))

        # 2️⃣ Signals
        if hasattr(self.brain, "generate_signals"):
            signals = self.brain.generate_signals(prices)
        elif hasattr(self.brain, "decide"):
            signals = self.brain.decide(prices)
        elif hasattr(self.brain, "predict"):
            signals = self.brain.predict(prices)
        elif hasattr(self.brain, "get_signals"):
            signals = self.brain.get_signals(prices)
        else:
            raise AttributeError("PortfolioBrain has no recognized signal-generation method")
        logger.info(f"🧠 Signals: {signals}")

        # 3️⃣ Risk pre-check
        if not self.guardian.validate(signals):
            logger.warning("🛑 RiskGuardian blocked trade cycle (pre-check)")
            return

        # 4️⃣ Execute trades
        for sym, sig in signals.items():
            side = (sig.get("side") or "FLAT").upper()
            qty = float(sig.get("qty") or 0.0)
            if side == "FLAT" or qty <= 0:
                continue
            order = await asyncio.to_thread(self._route_order_compat, sym, sig)
            if isinstance(order, dict):
                oid = self.tracker.track(order)
                await send_trade_alert(sym=sym, side=side, qty=qty, sig=sig, order=order)
                logger.info(f"📨 Routed {side} {qty}@{sym} → order_id={oid}")
            else:
                logger.error(f"Router returned non-dict for {sym}: {order}")

        # 5️⃣ Refresh & PnL
        self.tracker.refresh_with_router(self.router)
        pnl = self.guardian.evaluate_pnl(self.tracker.current_pnl())

        # breaker on hard drawdown
        if self.cooldown_on_dd and pnl < -self.hard_drawdown:
            self._enter_cooldown("drawdown_breaker")
            return

        self.telemetry.emit("pnl_snapshot", pnl)
        await send_pnl_alert(pnl)
        logger.info("✅ Cycle complete.\n")
