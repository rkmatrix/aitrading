"""
Order Validator for AITradeBot
-------------------------------
Comprehensive pre-trade validation to ensure orders are safe before execution.

Validates:
- Symbol format and validity
- Quantity bounds (min/max)
- Price sanity (not zero, not negative, within reasonable bounds)
- Buying power availability
- Market hours (if configured)
- Order type and parameters
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for order validation."""
    
    # Quantity limits
    min_qty: float = 1.0
    max_qty: float = 10000.0
    min_notional: float = 100.0  # Minimum order value in dollars
    
    # Price validation
    min_price: float = 0.01
    max_price: float = 100000.0  # Reasonable upper bound
    max_price_change_pct: float = 50.0  # Max price change from last known price
    
    # Buying power
    max_buying_power_usage_pct: float = 0.95  # Use max 95% of buying power
    
    # Symbol validation
    symbol_pattern: str = r"^[A-Z]{1,5}$"  # Basic stock symbol pattern
    allowed_symbols: Optional[list[str]] = None  # Whitelist if provided
    
    # Market hours (optional)
    check_market_hours: bool = False
    market_open_time: dtime = dtime(9, 30)  # 9:30 AM ET
    market_close_time: dtime = dtime(16, 0)  # 4:00 PM ET
    timezone: str = "America/New_York"
    
    # Order type validation
    allowed_order_types: list[str] = None  # Will default to ["market", "limit"]
    
    def __post_init__(self):
        if self.allowed_order_types is None:
            self.allowed_order_types = ["market", "limit"]


class OrderValidator:
    """
    Comprehensive order validator with pre-trade checks.
    
    Usage:
        validator = OrderValidator(config=ValidationConfig())
        is_valid, error_msg = validator.validate(order_dict, account_dict)
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger("OrderValidator")
        self._last_prices: Dict[str, float] = {}  # Cache for price change validation
        
    def validate(
        self,
        order: Dict[str, Any],
        account: Optional[Dict[str, Any]] = None,
        last_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Validate an order before execution.
        
        Args:
            order: Order dictionary with keys: symbol, side, qty, order_type, limit_price (optional)
            account: Account dictionary with keys: buying_power, equity (optional)
            last_price: Last known price for the symbol (optional, for price change validation)
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        # Extract order fields
        symbol = order.get("symbol", "").strip().upper()
        side = order.get("side", "").strip().upper()
        qty = order.get("qty")
        order_type = order.get("order_type", "market").strip().lower()
        limit_price = order.get("limit_price")
        price = last_price or order.get("price")
        
        # 1. Symbol validation
        is_valid, error = self._validate_symbol(symbol)
        if not is_valid:
            return False, error
        
        # 2. Side validation
        is_valid, error = self._validate_side(side)
        if not is_valid:
            return False, error
        
        # 3. Quantity validation
        is_valid, error = self._validate_quantity(qty)
        if not is_valid:
            return False, error
        
        # 4. Order type validation
        is_valid, error = self._validate_order_type(order_type, limit_price)
        if not is_valid:
            return False, error
        
        # 5. Price validation
        if price is not None:
            is_valid, error = self._validate_price(price, symbol)
            if not is_valid:
                return False, error
        
        # 6. Limit price validation (if limit order)
        if order_type == "limit" and limit_price is not None:
            is_valid, error = self._validate_limit_price(limit_price, side, price)
            if not is_valid:
                return False, error
        
        # 7. Notional value validation
        if price is not None:
            is_valid, error = self._validate_notional(qty, price)
            if not is_valid:
                return False, error
        
        # 8. Buying power validation (if account info provided)
        if account is not None:
            is_valid, error = self._validate_buying_power(side, qty, price, account)
            if not is_valid:
                return False, error
        
        # 9. Market hours validation (if enabled)
        if self.config.check_market_hours:
            is_valid, error = self._validate_market_hours()
            if not is_valid:
                return False, error
        
        # Cache last price for future validations
        if price is not None:
            self._last_prices[symbol] = price
        
        return True, "OK"
    
    def _validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate symbol format."""
        if not symbol:
            return False, "Symbol is required"
        
        # Check pattern
        if not re.match(self.config.symbol_pattern, symbol):
            return False, f"Invalid symbol format: {symbol}"
        
        # Check whitelist if provided
        if self.config.allowed_symbols and symbol not in self.config.allowed_symbols:
            return False, f"Symbol {symbol} not in allowed list"
        
        return True, "OK"
    
    def _validate_side(self, side: str) -> Tuple[bool, str]:
        """Validate order side."""
        if side not in ("BUY", "SELL", "buy", "sell"):
            return False, f"Invalid side: {side}. Must be BUY or SELL"
        return True, "OK"
    
    def _validate_quantity(self, qty: Any) -> Tuple[bool, str]:
        """Validate quantity."""
        if qty is None:
            return False, "Quantity is required"
        
        try:
            qty_float = float(qty)
        except (ValueError, TypeError):
            return False, f"Invalid quantity type: {type(qty)}"
        
        if qty_float <= 0:
            return False, f"Quantity must be positive, got {qty_float}"
        
        if qty_float < self.config.min_qty:
            return False, f"Quantity {qty_float} below minimum {self.config.min_qty}"
        
        if qty_float > self.config.max_qty:
            return False, f"Quantity {qty_float} exceeds maximum {self.config.max_qty}"
        
        # Check if whole number (for stocks)
        if abs(qty_float - round(qty_float)) > 1e-6:
            return False, f"Quantity must be a whole number for stocks, got {qty_float}"
        
        return True, "OK"
    
    def _validate_order_type(self, order_type: str, limit_price: Optional[float]) -> Tuple[bool, str]:
        """Validate order type."""
        if order_type not in self.config.allowed_order_types:
            return False, f"Invalid order type: {order_type}. Allowed: {self.config.allowed_order_types}"
        
        if order_type == "limit" and limit_price is None:
            return False, "Limit order requires limit_price"
        
        return True, "OK"
    
    def _validate_price(self, price: float, symbol: str) -> Tuple[bool, str]:
        """Validate price is within reasonable bounds."""
        if price <= 0:
            return False, f"Price must be positive, got {price}"
        
        if price < self.config.min_price:
            return False, f"Price {price} below minimum {self.config.min_price}"
        
        if price > self.config.max_price:
            return False, f"Price {price} exceeds maximum {self.config.max_price}"
        
        # Check price change from last known price
        if symbol in self._last_prices:
            last_price = self._last_prices[symbol]
            if last_price > 0:
                change_pct = abs((price - last_price) / last_price) * 100
                if change_pct > self.config.max_price_change_pct:
                    return False, (
                        f"Price change {change_pct:.2f}% exceeds maximum "
                        f"{self.config.max_price_change_pct}% (last: {last_price}, current: {price})"
                    )
        
        return True, "OK"
    
    def _validate_limit_price(self, limit_price: float, side: str, market_price: Optional[float]) -> Tuple[bool, str]:
        """Validate limit price is reasonable."""
        is_valid, error = self._validate_price(limit_price, "")
        if not is_valid:
            return False, f"Invalid limit price: {error}"
        
        # If we have market price, check limit is reasonable
        if market_price is not None and market_price > 0:
            if side.upper() == "BUY" and limit_price > market_price * 1.1:
                return False, f"Buy limit price {limit_price} too far above market {market_price}"
            if side.upper() == "SELL" and limit_price < market_price * 0.9:
                return False, f"Sell limit price {limit_price} too far below market {market_price}"
        
        return True, "OK"
    
    def _validate_notional(self, qty: float, price: float) -> Tuple[bool, str]:
        """Validate order notional value."""
        notional = abs(qty * price)
        if notional < self.config.min_notional:
            return False, (
                f"Order notional ${notional:.2f} below minimum ${self.config.min_notional}"
            )
        return True, "OK"
    
    def _validate_buying_power(
        self,
        side: str,
        qty: float,
        price: Optional[float],
        account: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Validate sufficient buying power for buy orders."""
        if side.upper() != "BUY" or price is None:
            return True, "OK"  # Only check buying power for buy orders with price
        
        buying_power = account.get("buying_power")
        if buying_power is None:
            return True, "OK"  # Skip if buying power not available
        
        try:
            buying_power_float = float(buying_power)
        except (ValueError, TypeError):
            return True, "OK"  # Skip if invalid buying power value
        
        required = qty * price
        max_allowed = buying_power_float * self.config.max_buying_power_usage_pct
        
        if required > max_allowed:
            return False, (
                f"Insufficient buying power: required ${required:.2f}, "
                f"available ${max_allowed:.2f} ({self.config.max_buying_power_usage_pct*100:.0f}% of ${buying_power_float:.2f})"
            )
        
        return True, "OK"
    
    def _validate_market_hours(self) -> Tuple[bool, str]:
        """Validate current time is within market hours."""
        try:
            tz = ZoneInfo(self.config.timezone)
            now = datetime.now(tz)
            current_time = now.time()
            
            # Check if weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False, f"Market is closed (weekend): {now.strftime('%A')}"
            
            if current_time < self.config.market_open_time:
                return False, (
                    f"Market is closed (before open): "
                    f"current {current_time.strftime('%H:%M')}, "
                    f"opens {self.config.market_open_time.strftime('%H:%M')}"
                )
            
            if current_time > self.config.market_close_time:
                return False, (
                    f"Market is closed (after close): "
                    f"current {current_time.strftime('%H:%M')}, "
                    f"closed {self.config.market_close_time.strftime('%H:%M')}"
                )
            
            return True, "OK"
        except Exception as e:
            self.logger.warning("Failed to validate market hours: %s", e)
            return True, "OK"  # Don't block on market hours check failure
    
    def update_last_price(self, symbol: str, price: float) -> None:
        """Update cached last price for a symbol."""
        self._last_prices[symbol] = price
