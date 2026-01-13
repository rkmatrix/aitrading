"""
Market Data Validator
---------------------
Validates market data quality before use in trading decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MarketDataConfig:
    """Configuration for market data validation."""
    
    # Staleness thresholds
    max_staleness_sec: float = 5.0  # Max age for real-time data
    max_staleness_warning_sec: float = 2.0  # Warn if data older than this
    
    # Price validation
    min_price: float = 0.01
    max_price: float = 100000.0
    max_price_change_pct: float = 50.0  # Max change from last known price
    
    # Spread validation
    max_spread_bps: float = 500.0  # Max bid-ask spread in basis points
    min_spread_bps: float = 0.0  # Min spread (negative spreads are invalid)
    
    # Volume validation
    min_volume: float = 0.0  # Minimum volume (0 = no check)
    
    # Data quality flags
    require_bid_ask: bool = True  # Require both bid and ask
    require_timestamp: bool = True  # Require timestamp


class MarketDataValidator:
    """
    Validates market data quality.
    
    Checks:
    - Data staleness
    - Price reasonableness
    - Spread validity
    - Required fields presence
    """
    
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        self.logger = logging.getLogger("MarketDataValidator")
        self._last_prices: Dict[str, Dict[str, Any]] = {}  # symbol -> {price, timestamp}
    
    def validate(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate market data.
        
        Args:
            symbol: Symbol being validated
            data: Market data dictionary with keys: bid, ask, last, timestamp, volume, etc.
        
        Returns:
            Tuple of (is_valid: bool, error_message: str, validated_data: dict)
        """
        validated = {}
        
        # 1. Check required fields
        is_valid, error = self._check_required_fields(data)
        if not is_valid:
            return False, error, {}
        
        # 2. Extract and validate timestamp
        timestamp = self._extract_timestamp(data)
        if timestamp is None:
            if self.config.require_timestamp:
                return False, "Missing timestamp", {}
        else:
            validated["timestamp"] = timestamp
            
            # Check staleness
            is_valid, error = self._check_staleness(symbol, timestamp)
            if not is_valid:
                return False, error, validated
        
        # 3. Validate prices
        bid = data.get("bid")
        ask = data.get("ask")
        last = data.get("last") or data.get("price")
        
        if bid is not None:
            is_valid, error = self._validate_price(bid, f"{symbol} bid")
            if not is_valid:
                return False, error, validated
            validated["bid"] = float(bid)
        
        if ask is not None:
            is_valid, error = self._validate_price(ask, f"{symbol} ask")
            if not is_valid:
                return False, error, validated
            validated["ask"] = float(ask)
        
        if last is not None:
            is_valid, error = self._validate_price(last, f"{symbol} last")
            if not is_valid:
                return False, error, validated
            validated["last"] = float(last)
            
            # Check price change from last known
            is_valid, error = self._check_price_change(symbol, float(last))
            if not is_valid:
                return False, error, validated
        
        # 4. Validate spread if both bid and ask present
        if bid is not None and ask is not None:
            is_valid, error = self._validate_spread(float(bid), float(ask))
            if not is_valid:
                return False, error, validated
            validated["spread_bps"] = ((float(ask) - float(bid)) / float(bid)) * 10000
        
        # 5. Validate volume if present
        volume = data.get("volume")
        if volume is not None:
            try:
                volume_float = float(volume)
                if volume_float < self.config.min_volume:
                    return False, f"Volume {volume_float} below minimum {self.config.min_volume}", validated
                validated["volume"] = volume_float
            except (ValueError, TypeError):
                pass  # Skip volume validation if invalid
        
        # Update cache
        if last is not None:
            self._last_prices[symbol] = {
                "price": float(last),
                "timestamp": timestamp or time.time(),
            }
        
        # Copy other fields
        for key in ["symbol", "exchange", "market_status"]:
            if key in data:
                validated[key] = data[key]
        
        validated["symbol"] = symbol
        validated["stale"] = False
        
        return True, "OK", validated
    
    def _check_required_fields(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check that required fields are present."""
        if self.config.require_bid_ask:
            if "bid" not in data and "ask" not in data:
                return False, "Missing both bid and ask prices"
        
        # At least one price field should be present
        if "bid" not in data and "ask" not in data and "last" not in data and "price" not in data:
            return False, "No price data available"
        
        return True, "OK"
    
    def _extract_timestamp(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract timestamp from data."""
        # Try various timestamp fields
        for key in ["timestamp", "ts", "time", "datetime"]:
            if key in data:
                ts = data[key]
                try:
                    if isinstance(ts, (int, float)):
                        return float(ts)
                    elif isinstance(ts, str):
                        # Try parsing ISO format
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        return dt.timestamp()
                except Exception:
                    continue
        
        return None
    
    def _check_staleness(self, symbol: str, timestamp: float) -> Tuple[bool, str]:
        """Check if data is stale."""
        now = time.time()
        age = now - timestamp
        
        if age > self.config.max_staleness_sec:
            return False, f"Data stale: {age:.2f}s > {self.config.max_staleness_sec}s"
        
        if age > self.config.max_staleness_warning_sec:
            self.logger.warning(
                "⚠️ Stale data for %s: %.2fs old (threshold: %.2fs)",
                symbol,
                age,
                self.config.max_staleness_warning_sec
            )
        
        return True, "OK"
    
    def _validate_price(self, price: Any, label: str) -> Tuple[bool, str]:
        """Validate price is reasonable."""
        try:
            price_float = float(price)
        except (ValueError, TypeError):
            return False, f"Invalid price type for {label}: {type(price)}"
        
        if price_float <= 0:
            return False, f"Price must be positive for {label}: {price_float}"
        
        if price_float < self.config.min_price:
            return False, f"Price too low for {label}: {price_float} < {self.config.min_price}"
        
        if price_float > self.config.max_price:
            return False, f"Price too high for {label}: {price_float} > {self.config.max_price}"
        
        return True, "OK"
    
    def _check_price_change(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """Check if price change is reasonable."""
        if symbol not in self._last_prices:
            return True, "OK"  # No previous price to compare
        
        last_data = self._last_prices[symbol]
        last_price = last_data["price"]
        
        if last_price <= 0:
            return True, "OK"  # Invalid last price, skip check
        
        change_pct = abs((current_price - last_price) / last_price) * 100
        
        if change_pct > self.config.max_price_change_pct:
            return False, (
                f"Price change too large: {change_pct:.2f}% "
                f"(last: {last_price}, current: {current_price})"
            )
        
        return True, "OK"
    
    def _validate_spread(self, bid: float, ask: float) -> Tuple[bool, str]:
        """Validate bid-ask spread."""
        if bid <= 0 or ask <= 0:
            return False, "Bid and ask must be positive"
        
        if ask < bid:
            return False, f"Invalid spread: ask {ask} < bid {bid}"
        
        spread_bps = ((ask - bid) / bid) * 10000
        
        if spread_bps < self.config.min_spread_bps:
            return False, f"Spread too narrow: {spread_bps:.2f}bps < {self.config.min_spread_bps}bps"
        
        if spread_bps > self.config.max_spread_bps:
            return False, f"Spread too wide: {spread_bps:.2f}bps > {self.config.max_spread_bps}bps"
        
        return True, "OK"
