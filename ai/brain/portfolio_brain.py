"""
PortfolioBrain – Core portfolio intelligence layer
(Used by Phase 30 Allocator and Phase 31 Risk Controller)

Responsibilities:
    • Track symbol exposures and equity curve
    • Simulate trades / portfolio updates
    • Provide drawdown and volatility metrics
    • Support scaling and freezing for risk control
"""

from __future__ import annotations
import numpy as np
import logging, time

logger = logging.getLogger(__name__)


class PortfolioBrain:
    def __init__(self, symbols: list[str], starting_equity: float = 100000.0):
        self.symbols = symbols
        self.equity = starting_equity
        self.positions = {s: 0.0 for s in symbols}
        self.prices = {s: 100.0 for s in symbols}
        self.trade_history = []
        self.frozen = False
        logger.info("🧠 PortfolioBrain initialized (equity=%.2f, symbols=%s)", self.equity, symbols)

    # ------------------------------------------------------------------
    # Basic Simulated Methods
    # ------------------------------------------------------------------
    def update_prices(self, new_prices: dict[str, float]):
        """Update internal price dictionary."""
        self.prices.update(new_prices)

    def place_order(self, symbol: str, qty: float, price: float | None = None):
        """Simulate placing a buy/sell order and updating position."""
        if self.frozen:
            logger.warning("🚫 Trade blocked – PortfolioBrain is frozen by risk controller.")
            return False

        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")

        price = price or self.prices.get(symbol, 100.0)
        cost = qty * price
        self.equity -= cost
        self.positions[symbol] += qty
        self.trade_history.append({
            "t": time.time(),
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "cost": cost,
        })
        logger.info("🟢 Order executed: %s %+d @ %.2f", symbol, qty, price)
        return True

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_current_exposures(self) -> dict[str, float]:
        """
        Exposure fraction per symbol = |value of position| / total portfolio value.
        """
        total_val = self.get_portfolio_value()
        exposures = {}
        for sym, qty in self.positions.items():
            val = abs(qty * self.prices[sym])
            exposures[sym] = float(val / total_val) if total_val > 0 else 0.0
        return exposures

    def get_portfolio_value(self) -> float:
        """Compute total portfolio market value."""
        val = self.equity
        for s, qty in self.positions.items():
            val += qty * self.prices[s]
        return float(val)

    def get_drawdown(self, window: int = 30) -> float:
        """
        Compute a simple rolling drawdown percentage based on trade history equity curve.
        """
        if not self.trade_history:
            return 0.0
        equity_curve = np.cumsum([t["cost"] for t in self.trade_history])
        peak = np.max(equity_curve)
        trough = equity_curve[-1]
        dd = (peak - trough) / (peak + 1e-9)
        return float(max(0.0, min(dd, 1.0)))

    # ------------------------------------------------------------------
    # Risk-control hooks
    # ------------------------------------------------------------------
    def freeze_trading(self):
        """Freeze further trading activity."""
        self.frozen = True
        logger.warning("🛑 PortfolioBrain trading frozen due to risk breach.")

    def unfreeze(self):
        self.frozen = False
        logger.info("✅ PortfolioBrain trading resumed.")

    def scale_positions(self, factor: float):
        """
        Reduce all position sizes by given factor (e.g., 0.8 = cut by 20%).
        """
        for s in self.symbols:
            self.positions[s] *= factor
        logger.warning("📉 Portfolio positions scaled by factor %.2f", factor)
