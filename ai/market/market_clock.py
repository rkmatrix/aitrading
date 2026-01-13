"""
Market Clock - Real Exchange Hours Implementation
-------------------------------------------------
Provides accurate market hours checking using exchange calendars.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
import pytz

logger = logging.getLogger(__name__)


class MarketClock:
    """
    Real market hours checker using exchange calendar.
    
    Supports:
    - Regular trading hours (9:30 AM - 4:00 PM ET)
    - Pre-market hours (4:00 AM - 9:30 AM ET)
    - After-hours (4:00 PM - 8:00 PM ET)
    - Weekends and holidays
    """
    
    # Market hours (Eastern Time)
    REGULAR_OPEN = dtime(9, 30)   # 9:30 AM ET
    REGULAR_CLOSE = dtime(16, 0)  # 4:00 PM ET
    PRE_MARKET_OPEN = dtime(4, 0)  # 4:00 AM ET
    AFTER_HOURS_CLOSE = dtime(20, 0)  # 8:00 PM ET
    
    # Exchange timezone
    EXCHANGE_TZ = ZoneInfo("America/New_York")
    
    def __init__(self, check_premarket: bool = False, check_afterhours: bool = False):
        """
        Initialize market clock.
        
        Args:
            check_premarket: If True, consider pre-market hours as "open"
            check_afterhours: If True, consider after-hours as "open"
        """
        self.check_premarket = check_premarket
        self.check_afterhours = check_afterhours
        self.logger = logging.getLogger("MarketClock")
        
        # Cache for holiday checking (can be extended with actual holiday calendar)
        self._holidays_cache: Optional[set] = None
    
    def is_open(self, check_extended: Optional[bool] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            check_extended: Override instance settings for pre/after hours
        
        Returns:
            True if market is open, False otherwise
        """
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if holiday (simplified - can be enhanced with actual calendar)
        if self._is_holiday(now_et):
            return False
        
        current_time = now_et.time()
        
        # Determine if we should check extended hours
        check_pre = check_extended if check_extended is not None else self.check_premarket
        check_after = check_extended if check_extended is not None else self.check_afterhours
        
        # Regular trading hours
        if self.REGULAR_OPEN <= current_time <= self.REGULAR_CLOSE:
            return True
        
        # Pre-market hours
        if check_pre and self.PRE_MARKET_OPEN <= current_time < self.REGULAR_OPEN:
            return True
        
        # After-hours
        if check_after and self.REGULAR_CLOSE < current_time <= self.AFTER_HOURS_CLOSE:
            return True
        
        return False
    
    def is_regular_hours(self) -> bool:
        """Check if currently in regular trading hours (9:30 AM - 4:00 PM ET)."""
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        if now_et.weekday() >= 5:
            return False
        
        if self._is_holiday(now_et):
            return False
        
        current_time = now_et.time()
        return self.REGULAR_OPEN <= current_time <= self.REGULAR_CLOSE
    
    def is_premarket(self) -> bool:
        """Check if currently in pre-market hours."""
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        if now_et.weekday() >= 5:
            return False
        
        if self._is_holiday(now_et):
            return False
        
        current_time = now_et.time()
        return self.PRE_MARKET_OPEN <= current_time < self.REGULAR_OPEN
    
    def is_afterhours(self) -> bool:
        """Check if currently in after-hours."""
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        if now_et.weekday() >= 5:
            return False
        
        if self._is_holiday(now_et):
            return False
        
        current_time = now_et.time()
        return self.REGULAR_CLOSE < current_time <= self.AFTER_HOURS_CLOSE
    
    def time_until_open(self) -> Optional[float]:
        """
        Get seconds until market opens (regular hours).
        
        Returns:
            Seconds until open, or None if already open or on holiday/weekend
        """
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        # If already open, return 0
        if self.is_regular_hours():
            return 0.0
        
        # If weekend, calculate to next Monday
        if now_et.weekday() >= 5:
            days_until_monday = (7 - now_et.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open = next_open.replace(day=now_et.day + days_until_monday)
        else:
            # Same day if before open, next day if after close
            if now_et.time() < self.REGULAR_OPEN:
                next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                # Next trading day
                next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                next_open = next_open.replace(day=now_et.day + 1)
        
        delta = (next_open - now_et).total_seconds()
        return delta if delta > 0 else None
    
    def time_until_close(self) -> Optional[float]:
        """
        Get seconds until market closes (regular hours).
        
        Returns:
            Seconds until close, or None if already closed
        """
        if not self.is_regular_hours():
            return None
        
        now_et = datetime.now(self.EXCHANGE_TZ)
        today_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        delta = (today_close - now_et).total_seconds()
        return delta if delta > 0 else None
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get comprehensive market status.
        
        Returns:
            Dict with market status information
        """
        now_et = datetime.now(self.EXCHANGE_TZ)
        
        return {
            "is_open": self.is_open(),
            "is_regular_hours": self.is_regular_hours(),
            "is_premarket": self.is_premarket(),
            "is_afterhours": self.is_afterhours(),
            "is_weekend": now_et.weekday() >= 5,
            "is_holiday": self._is_holiday(now_et),
            "current_time_et": now_et.isoformat(),
            "time_until_open": self.time_until_open(),
            "time_until_close": self.time_until_close(),
        }
    
    def _is_holiday(self, date: datetime) -> bool:
        """
        Check if date is a market holiday.
        
        Note: This is a simplified implementation. For production use,
        integrate with a proper holiday calendar library like pandas_market_calendars.
        """
        # Major US market holidays (simplified list)
        # In production, use: pandas_market_calendars.get_calendar('NYSE')
        major_holidays = {
            # New Year's Day
            (1, 1),
            # Independence Day
            (7, 4),
            # Christmas
            (12, 25),
            # Thanksgiving (4th Thursday of November - simplified)
            # Note: This is approximate
        }
        
        month_day = (date.month, date.day)
        if month_day in major_holidays:
            return True
        
        # Check for Monday holidays (if holiday falls on weekend)
        # This is simplified - real implementation should use proper calendar
        
        return False
    
    @classmethod
    def create_with_calendar(cls, calendar_name: str = "NYSE") -> "MarketClock":
        """
        Create MarketClock with actual exchange calendar.
        
        Requires: pip install pandas_market_calendars
        
        Args:
            calendar_name: Exchange calendar name (e.g., 'NYSE', 'NASDAQ')
        
        Returns:
            MarketClock instance with calendar integration
        """
        try:
            import pandas_market_calendars as mcal
            
            calendar = mcal.get_calendar(calendar_name)
            
            # Create enhanced clock with calendar
            clock = cls()
            clock._calendar = calendar
            clock._use_calendar = True
            
            logger.info("MarketClock initialized with %s calendar", calendar_name)
            return clock
            
        except ImportError:
            logger.warning(
                "pandas_market_calendars not installed. Using simplified holiday checking. "
                "Install with: pip install pandas_market_calendars"
            )
            return cls()
        except Exception as e:
            logger.warning("Failed to load calendar: %s. Using simplified checking.", e)
            return cls()
