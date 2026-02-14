"""
Box Trading Strategy Module
Implements mean-reversion trading based on previous day's high/low range
with advanced safeguards, regime filtering, and risk management.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - some features may be limited")


class BoxLevels:
    """Container for box level data"""
    def __init__(
        self,
        symbol: str,
        prev_day_high: float,
        prev_day_low: float,
        prev_day_close: float,
        prev_day_volume: int,
        timestamp: datetime,
        is_alternative: bool = False
    ):
        self.symbol = symbol
        self.prev_day_high = prev_day_high
        self.prev_day_low = prev_day_low
        self.prev_day_close = prev_day_close
        self.prev_day_volume = prev_day_volume
        self.midpoint = (prev_day_high + prev_day_low) / 2
        self.range_size = prev_day_high - prev_day_low
        self.range_percent = (self.range_size / prev_day_low) if prev_day_low > 0 else 0
        self.timestamp = timestamp
        self.is_alternative = is_alternative
        self.boundary_touches = {"top": 0, "bottom": 0}
        
    def get_position_in_range(self, price: float) -> float:
        """Returns position in range from 0.0 (bottom) to 1.0 (top)"""
        if self.range_size == 0:
            return 0.5
        return (price - self.prev_day_low) / self.range_size
    
    def __repr__(self):
        return (f"BoxLevels({self.symbol}: "
                f"H={self.prev_day_high:.2f}, L={self.prev_day_low:.2f}, "
                f"Mid={self.midpoint:.2f}, Range={self.range_percent*100:.2f}%)")


class TradeSignal:
    """Container for trade signal with all context"""
    def __init__(
        self,
        symbol: str,
        action: str,  # "BUY", "SELL", "HOLD"
        confidence: float,
        current_price: float,
        box_levels: BoxLevels,
        reasons: List[str],
        stop_loss: float,
        take_profit_targets: List[float],
        risk_reward_ratio: float,
        timestamp: datetime,
        confirmations: Dict[str, Any]
    ):
        self.symbol = symbol
        self.action = action
        self.confidence = confidence
        self.current_price = current_price
        self.box_levels = box_levels
        self.reasons = reasons
        self.stop_loss = stop_loss
        self.take_profit_targets = take_profit_targets
        self.risk_reward_ratio = risk_reward_ratio
        self.timestamp = timestamp
        self.confirmations = confirmations
        
    def __repr__(self):
        return (f"TradeSignal({self.symbol} {self.action} @ {self.current_price:.2f}, "
                f"Conf={self.confidence:.2f}, R:R={self.risk_reward_ratio:.2f})")


class BoxTradingStrategy:
    """
    Main Box Trading Strategy Implementation
    Features:
    - Previous day high/low box calculation
    - Multi-layer regime filtering
    - Breakout detection and avoidance
    - Whipsaw protection
    - Adaptive zone thresholds
    - Correlation filtering
    - Performance-based learning
    """
    
    def __init__(self, config: Dict[str, Any], data_provider):
        self.config = config
        self.data_provider = data_provider
        
        # State tracking
        self.box_cache: Dict[str, BoxLevels] = {}
        self.symbol_blacklist: Dict[str, datetime] = {}
        self.recent_stops: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.boundary_touch_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {"top": deque(maxlen=20), "bottom": deque(maxlen=20)}
        )
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_by_symbol: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0}
        )
        self.performance_by_hour: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"wins": 0, "losses": 0}
        )
        
        logger.info("BoxTradingStrategy initialized")
    
    def calculate_box_levels(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
        force_recalculate: bool = False
    ) -> Optional[BoxLevels]:
        """
        Calculate box levels from previous day's high/low
        Returns None if insufficient data or invalid box
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check cache
        cache_key = f"{symbol}_{current_time.date()}"
        if not force_recalculate and cache_key in self.box_cache:
            return self.box_cache[cache_key]
        
        try:
            # Get daily historical data
            hist_data = self.data_provider.get_historical_data(
                symbol=symbol,
                period="5d",  # Get 5 days to ensure we have previous day
                interval="1d"
            )
            
            if not hist_data or len(hist_data) < 2:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Get previous day's data (second to last bar)
            prev_day = hist_data[-2]
            
            prev_day_high = prev_day.get("high", 0)
            prev_day_low = prev_day.get("low", 0)
            prev_day_close = prev_day.get("close", 0)
            prev_day_volume = prev_day.get("volume", 0)
            
            if prev_day_high <= 0 or prev_day_low <= 0:
                logger.warning(f"Invalid price data for {symbol}")
                return None
            
            # Create box levels
            box_levels = BoxLevels(
                symbol=symbol,
                prev_day_high=prev_day_high,
                prev_day_low=prev_day_low,
                prev_day_close=prev_day_close,
                prev_day_volume=prev_day_volume,
                timestamp=current_time
            )
            
            # Validate box (min range requirement)
            min_range = self.config.get("volatility_filter", {}).get("min_atr_percent", 0.005)
            if box_levels.range_percent < min_range:
                logger.debug(f"{symbol} box range too small: {box_levels.range_percent*100:.2f}%")
                return None
            
            # Cache it
            self.box_cache[cache_key] = box_levels
            logger.debug(f"Calculated box for {symbol}: {box_levels}")
            
            return box_levels
            
        except Exception as e:
            logger.error(f"Error calculating box levels for {symbol}: {e}")
            return None
    
    def calculate_alternative_box(
        self,
        symbol: str,
        current_session_bars: List[Dict[str, Any]],
        current_time: datetime
    ) -> Optional[BoxLevels]:
        """
        Calculate alternative box from current session high/low
        Used when price gaps significantly outside previous day's range
        """
        min_bars = self.config.get("min_bars_for_alt_box", 10)
        
        if len(current_session_bars) < min_bars:
            logger.debug(f"Not enough bars for alternative box: {len(current_session_bars)}/{min_bars}")
            return None
        
        try:
            highs = [bar["high"] for bar in current_session_bars]
            lows = [bar["low"] for bar in current_session_bars]
            closes = [bar["close"] for bar in current_session_bars]
            volumes = [bar["volume"] for bar in current_session_bars]
            
            session_high = max(highs)
            session_low = min(lows)
            session_close = closes[-1]
            session_volume = sum(volumes)
            
            alt_box = BoxLevels(
                symbol=symbol,
                prev_day_high=session_high,
                prev_day_low=session_low,
                prev_day_close=session_close,
                prev_day_volume=session_volume,
                timestamp=current_time,
                is_alternative=True
            )
            
            logger.info(f"Created alternative box for {symbol}: {alt_box}")
            return alt_box
            
        except Exception as e:
            logger.error(f"Error creating alternative box for {symbol}: {e}")
            return None
    
    def get_adaptive_zone_thresholds(
        self,
        symbol: str,
        box_levels: BoxLevels,
        current_atr_percent: float
    ) -> Tuple[float, float]:
        """
        Calculate adaptive zone thresholds based on volatility
        Returns (top_threshold, bottom_threshold)
        """
        mode = self.config.get("zone_calculation_mode", "adaptive")
        
        if mode == "fixed":
            top = self.config.get("fixed_top_zone_threshold", 0.005)
            bottom = self.config.get("fixed_bottom_zone_threshold", 0.005)
            return (top, bottom)
        
        # Adaptive mode
        adaptive_zones = self.config.get("adaptive_zones", {})
        
        if current_atr_percent < 0.01:  # Low volatility
            zone = adaptive_zones.get("low_volatility", {})
        elif current_atr_percent < 0.02:  # Medium volatility
            zone = adaptive_zones.get("medium_volatility", {})
        else:  # High volatility
            zone = adaptive_zones.get("high_volatility", {})
        
        top = zone.get("top_threshold", 0.007)
        bottom = zone.get("bottom_threshold", 0.007)
        
        logger.debug(f"{symbol} adaptive thresholds: top={top*100:.2f}%, bottom={bottom*100:.2f}% "
                    f"(ATR={current_atr_percent*100:.2f}%)")
        
        return (top, bottom)
    
    def is_in_zone(
        self,
        price: float,
        box_levels: BoxLevels,
        zone: str,  # "top", "bottom", "middle"
        top_threshold: float,
        bottom_threshold: float
    ) -> bool:
        """Check if price is in specified zone"""
        position = box_levels.get_position_in_range(price)
        middle_avoid = self.config.get("middle_zone_avoid", 0.40)
        
        if zone == "top":
            # Top zone: within threshold of the high
            return position >= (1.0 - top_threshold)
        elif zone == "bottom":
            # Bottom zone: within threshold of the low
            return position <= bottom_threshold
        elif zone == "middle":
            # Middle zone: the avoid zone in the center
            middle_start = 0.5 - (middle_avoid / 2)
            middle_end = 0.5 + (middle_avoid / 2)
            return middle_start <= position <= middle_end
        
        return False
    
    def detect_breakout(
        self,
        symbol: str,
        current_price: float,
        box_levels: BoxLevels,
        recent_bars: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect if symbol is breaking out (not mean-reverting)
        Returns True if breakout detected
        """
        breakout_config = self.config.get("breakout_detection", {})
        if not breakout_config.get("enabled", True):
            return False
        
        volume_threshold = breakout_config.get("volume_spike_threshold", 2.0)
        momentum_threshold = breakout_config.get("momentum_threshold", 0.8)
        consecutive_breaks = breakout_config.get("consecutive_breaks", 2)
        
        if len(recent_bars) < consecutive_breaks + 5:
            return False
        
        try:
            # Check for consecutive breaks outside box
            breaks_above = 0
            breaks_below = 0
            
            for bar in recent_bars[-consecutive_breaks:]:
                bar_high = bar.get("high", 0)
                bar_low = bar.get("low", 0)
                
                if bar_high > box_levels.prev_day_high:
                    breaks_above += 1
                if bar_low < box_levels.prev_day_low:
                    breaks_below += 1
            
            has_consecutive_break = (breaks_above >= consecutive_breaks or 
                                    breaks_below >= consecutive_breaks)
            
            if not has_consecutive_break:
                return False
            
            # Check volume spike
            recent_volume = np.mean([bar.get("volume", 0) for bar in recent_bars[-5:]])
            avg_volume = np.mean([bar.get("volume", 0) for bar in recent_bars[-20:-5]])
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                has_volume_spike = volume_ratio > volume_threshold
            else:
                has_volume_spike = False
            
            # Check momentum
            closes = [bar.get("close", 0) for bar in recent_bars[-10:]]
            if len(closes) >= 2:
                momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
                has_strong_momentum = abs(momentum) > momentum_threshold / 100
            else:
                has_strong_momentum = False
            
            # Breakout if 2 of 3 conditions met
            conditions_met = sum([has_consecutive_break, has_volume_spike, has_strong_momentum])
            
            if conditions_met >= 2:
                logger.warning(f"BREAKOUT DETECTED for {symbol}: "
                             f"consecutive_break={has_consecutive_break}, "
                             f"volume_spike={has_volume_spike}, "
                             f"momentum={has_strong_momentum}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting breakout for {symbol}: {e}")
            return False
    
    def is_symbol_blacklisted(self, symbol: str, current_time: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol is blacklisted
        Returns (is_blacklisted, reason)
        """
        if symbol in self.symbol_blacklist:
            blacklist_until = self.symbol_blacklist[symbol]
            if current_time < blacklist_until:
                remaining = (blacklist_until - current_time).total_seconds() / 60
                return (True, f"Blacklisted for {remaining:.0f} more minutes")
            else:
                # Blacklist expired
                del self.symbol_blacklist[symbol]
        
        return (False, None)
    
    def add_to_blacklist(
        self,
        symbol: str,
        duration_minutes: int,
        reason: str,
        current_time: Optional[datetime] = None
    ):
        """Add symbol to blacklist"""
        if current_time is None:
            current_time = datetime.now()
        
        blacklist_until = current_time + timedelta(minutes=duration_minutes)
        self.symbol_blacklist[symbol] = blacklist_until
        
        logger.warning(f"Blacklisted {symbol} until {blacklist_until} ({reason})")
    
    def check_whipsaw_protection(
        self,
        symbol: str,
        current_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol has been stopped out recently (whipsaw protection)
        Returns (should_skip, reason)
        """
        whipsaw_config = self.config.get("whipsaw_protection", {})
        if not whipsaw_config.get("enabled", True):
            return (False, None)
        
        max_stops = whipsaw_config.get("max_stops_per_symbol_per_hour", 2)
        lookback_minutes = 60
        blacklist_after = whipsaw_config.get("blacklist_after_stops", 2)
        blacklist_duration = whipsaw_config.get("blacklist_duration_minutes", 240)
        
        # Count recent stops
        recent_stop_times = [
            stop_time for stop_time in self.recent_stops[symbol]
            if (current_time - stop_time).total_seconds() / 60 <= lookback_minutes
        ]
        
        if len(recent_stop_times) >= max_stops:
            # Too many stops - blacklist
            self.add_to_blacklist(
                symbol,
                blacklist_duration,
                f"Too many stops: {len(recent_stop_times)} in {lookback_minutes}min",
                current_time
            )
            return (True, f"Whipsaw protection: {len(recent_stop_times)} stops in past hour")
        
        return (False, None)
    
    def record_stop_out(self, symbol: str, stop_time: Optional[datetime] = None):
        """Record that symbol was stopped out"""
        if stop_time is None:
            stop_time = datetime.now()
        
        self.recent_stops[symbol].append(stop_time)
        logger.info(f"Recorded stop out for {symbol} at {stop_time}")
    
    def count_boundary_touches(
        self,
        symbol: str,
        boundary_type: str,  # "top" or "bottom"
        current_time: datetime,
        lookback_bars: int = 20
    ) -> int:
        """
        Count how many times the boundary has been touched recently
        More touches = higher confidence in the level
        """
        touches = self.boundary_touch_history[symbol][boundary_type]
        
        # Count touches within lookback period
        recent_touches = [
            touch_time for touch_time in touches
            if (current_time - touch_time).total_seconds() / 60 <= lookback_bars * 5  # Assume 5min bars
        ]
        
        return len(recent_touches)
    
    def record_boundary_touch(
        self,
        symbol: str,
        boundary_type: str,
        touch_time: Optional[datetime] = None
    ):
        """Record that a boundary was touched"""
        if touch_time is None:
            touch_time = datetime.now()
        
        self.boundary_touch_history[symbol][boundary_type].append(touch_time)
    
    def check_confirmations(
        self,
        symbol: str,
        current_price: float,
        box_levels: BoxLevels,
        action: str,  # "BUY" or "SELL"
        recent_bars: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check all technical confirmations
        Returns (passed, confirmation_data, reasons)
        """
        confirmations_config = self.config.get("confirmations", {})
        confirmation_data = {}
        reasons = []
        passed = True
        
        if len(recent_bars) < 20:
            return (False, {}, ["Insufficient bar data for confirmations"])
        
        try:
            # RSI confirmation
            if confirmations_config.get("use_rsi", True):
                closes = [bar.get("close", 0) for bar in recent_bars[-20:]]
                rsi = self._calculate_rsi(closes, period=confirmations_config.get("rsi_period", 14))
                confirmation_data["rsi"] = rsi
                
                rsi_oversold = confirmations_config.get("rsi_oversold", 30)
                rsi_overbought = confirmations_config.get("rsi_overbought", 70)
                
                if action == "BUY":
                    if rsi < rsi_oversold:
                        reasons.append(f"RSI oversold ({rsi:.1f})")
                    else:
                        passed = False
                        reasons.append(f"RSI not oversold ({rsi:.1f} > {rsi_oversold})")
                
                elif action == "SELL":
                    if rsi > rsi_overbought:
                        reasons.append(f"RSI overbought ({rsi:.1f})")
                    else:
                        passed = False
                        reasons.append(f"RSI not overbought ({rsi:.1f} < {rsi_overbought})")
            
            # Volume confirmation
            if confirmations_config.get("use_volume", True):
                recent_volume = recent_bars[-1].get("volume", 0)
                avg_volume = np.mean([bar.get("volume", 0) for bar in recent_bars[-20:]])
                
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    confirmation_data["volume_ratio"] = volume_ratio
                    
                    volume_threshold = confirmations_config.get("volume_threshold", 1.5)
                    
                    if volume_ratio >= volume_threshold:
                        reasons.append(f"Volume confirmation ({volume_ratio:.2f}x avg)")
                    else:
                        passed = False
                        reasons.append(f"Low volume ({volume_ratio:.2f}x < {volume_threshold}x)")
            
            # Rejection candle check
            if confirmations_config.get("require_rejection_candle", False):
                last_bar = recent_bars[-1]
                bar_open = last_bar.get("open", 0)
                bar_close = last_bar.get("close", 0)
                bar_high = last_bar.get("high", 0)
                bar_low = last_bar.get("low", 0)
                
                bar_range = bar_high - bar_low
                if bar_range > 0:
                    min_wick_ratio = confirmations_config.get("min_rejection_wick_ratio", 0.4)
                    
                    if action == "BUY":
                        # Need bearish rejection (long lower wick)
                        lower_wick = min(bar_open, bar_close) - bar_low
                        wick_ratio = lower_wick / bar_range
                        
                        if wick_ratio >= min_wick_ratio:
                            reasons.append(f"Rejection candle (wick {wick_ratio*100:.0f}%)")
                        else:
                            passed = False
                            reasons.append(f"No rejection candle (wick {wick_ratio*100:.0f}%)")
                    
                    elif action == "SELL":
                        # Need bullish rejection (long upper wick)
                        upper_wick = bar_high - max(bar_open, bar_close)
                        wick_ratio = upper_wick / bar_range
                        
                        if wick_ratio >= min_wick_ratio:
                            reasons.append(f"Rejection candle (wick {wick_ratio*100:.0f}%)")
                        else:
                            passed = False
                            reasons.append(f"No rejection candle (wick {wick_ratio*100:.0f}%)")
            
            # Momentum check (don't fade strong momentum)
            if confirmations_config.get("check_momentum", True):
                closes = [bar.get("close", 0) for bar in recent_bars[-10:]]
                if len(closes) >= 2 and closes[0] > 0:
                    momentum = (closes[-1] - closes[0]) / closes[0]
                    confirmation_data["momentum"] = momentum
                    
                    max_momentum = confirmations_config.get("max_momentum_for_entry", 0.5) / 100
                    
                    if action == "BUY" and momentum < -max_momentum:
                        # Strong downward momentum - might be breakout, not mean reversion
                        passed = False
                        reasons.append(f"Too much downward momentum ({momentum*100:.2f}%)")
                    elif action == "SELL" and momentum > max_momentum:
                        # Strong upward momentum - might be breakout
                        passed = False
                        reasons.append(f"Too much upward momentum ({momentum*100:.2f}%)")
            
            return (passed, confirmation_data, reasons)
            
        except Exception as e:
            logger.error(f"Error checking confirmations for {symbol}: {e}")
            return (False, {}, [f"Error in confirmations: {e}"])
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(closes) < period + 1:
            return 50.0  # Neutral
        
        try:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        current_time: Optional[datetime] = None,
        recent_bars: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[TradeSignal]:
        """
        Main signal generation function
        Returns TradeSignal or None
        """
        if current_time is None:
            current_time = datetime.now()
        
        try:
            # Check blacklist
            is_blacklisted, blacklist_reason = self.is_symbol_blacklisted(symbol, current_time)
            if is_blacklisted:
                logger.debug(f"{symbol} skipped: {blacklist_reason}")
                return None
            
            # Check whipsaw protection
            should_skip, whipsaw_reason = self.check_whipsaw_protection(symbol, current_time)
            if should_skip:
                logger.debug(f"{symbol} skipped: {whipsaw_reason}")
                return None
            
            # Calculate box levels
            box_levels = self.calculate_box_levels(symbol, current_time)
            if box_levels is None:
                return None
            
            # Check for gap and alternative box
            gap_threshold = self.config.get("gap_threshold", 0.02)
            if current_price > box_levels.prev_day_high * (1 + gap_threshold):
                logger.info(f"{symbol} gapped up above box, evaluating alternative box")
                if recent_bars and len(recent_bars) >= self.config.get("min_bars_for_alt_box", 10):
                    alt_box = self.calculate_alternative_box(symbol, recent_bars, current_time)
                    if alt_box:
                        box_levels = alt_box
                    else:
                        logger.debug(f"{symbol} gap up but can't create alt box yet")
                        return None
                else:
                    return None
            
            elif current_price < box_levels.prev_day_low * (1 - gap_threshold):
                logger.info(f"{symbol} gapped down below box, evaluating alternative box")
                if recent_bars and len(recent_bars) >= self.config.get("min_bars_for_alt_box", 10):
                    alt_box = self.calculate_alternative_box(symbol, recent_bars, current_time)
                    if alt_box:
                        box_levels = alt_box
                    else:
                        logger.debug(f"{symbol} gap down but can't create alt box yet")
                        return None
                else:
                    return None
            
            # Check for breakout
            if recent_bars and self.detect_breakout(symbol, current_price, box_levels, recent_bars):
                # Blacklist symbol after breakout
                blacklist_duration = self.config.get("breakout_detection", {}).get(
                    "blacklist_duration_minutes", 120
                )
                self.add_to_blacklist(symbol, blacklist_duration, "Breakout detected", current_time)
                return None
            
            # Get adaptive zone thresholds
            # Estimate ATR from box range
            current_atr_percent = box_levels.range_percent
            top_threshold, bottom_threshold = self.get_adaptive_zone_thresholds(
                symbol, box_levels, current_atr_percent
            )
            
            # Determine which zone we're in
            in_top_zone = self.is_in_zone(current_price, box_levels, "top", top_threshold, bottom_threshold)
            in_bottom_zone = self.is_in_zone(current_price, box_levels, "bottom", top_threshold, bottom_threshold)
            in_middle_zone = self.is_in_zone(current_price, box_levels, "middle", top_threshold, bottom_threshold)
            
            # Don't trade in middle zone
            if in_middle_zone:
                logger.debug(f"{symbol} in middle zone, skipping")
                return None
            
            # Determine action
            action = "HOLD"
            base_confidence = 0.5
            reasons = []
            
            if in_bottom_zone:
                action = "BUY"
                reasons.append(f"Price in bottom zone ({box_levels.get_position_in_range(current_price)*100:.1f}%)")
                
                # Record boundary touch
                self.record_boundary_touch(symbol, "bottom", current_time)
                
                # Increase confidence based on touches
                touches = self.count_boundary_touches(symbol, "bottom", current_time)
                if touches == 0:
                    base_confidence = 0.70
                    reasons.append("First touch of bottom")
                elif touches == 1:
                    base_confidence = 0.80
                    reasons.append("Second touch of bottom")
                else:
                    base_confidence = 0.90
                    reasons.append(f"Proven bottom ({touches+1} touches)")
            
            elif in_top_zone:
                action = "SELL"
                reasons.append(f"Price in top zone ({box_levels.get_position_in_range(current_price)*100:.1f}%)")
                
                # Record boundary touch
                self.record_boundary_touch(symbol, "top", current_time)
                
                # Increase confidence based on touches
                touches = self.count_boundary_touches(symbol, "top", current_time)
                if touches == 0:
                    base_confidence = 0.70
                    reasons.append("First touch of top")
                elif touches == 1:
                    base_confidence = 0.80
                    reasons.append("Second touch of top")
                else:
                    base_confidence = 0.90
                    reasons.append(f"Proven top ({touches+1} touches)")
            
            else:
                # Not in any zone
                return None
            
            # Check confirmations
            if recent_bars:
                confirmations_passed, confirmation_data, confirmation_reasons = self.check_confirmations(
                    symbol, current_price, box_levels, action, recent_bars
                )
                
                reasons.extend(confirmation_reasons)
                
                if not confirmations_passed:
                    logger.debug(f"{symbol} failed confirmations: {confirmation_reasons}")
                    return None
                
                # Boost confidence if confirmations strong
                if confirmation_data.get("rsi", 50) < 25 or confirmation_data.get("rsi", 50) > 75:
                    base_confidence += 0.05
                if confirmation_data.get("volume_ratio", 1.0) > 2.0:
                    base_confidence += 0.05
            else:
                confirmation_data = {}
            
            # Cap confidence
            confidence = min(base_confidence, 1.0)
            
            # Calculate stop loss and take profit
            stop_loss_buffer = self.config.get("stop_loss_buffer", 0.003)
            
            if action == "BUY":
                stop_loss = box_levels.prev_day_low * (1 - stop_loss_buffer)
                tier1_target = box_levels.midpoint
                tier2_target = box_levels.prev_day_low + (box_levels.range_size * 0.75)
                tier3_target = box_levels.prev_day_high * (1 - stop_loss_buffer)
            else:  # SELL
                stop_loss = box_levels.prev_day_high * (1 + stop_loss_buffer)
                tier1_target = box_levels.midpoint
                tier2_target = box_levels.prev_day_high - (box_levels.range_size * 0.75)
                tier3_target = box_levels.prev_day_low * (1 + stop_loss_buffer)
            
            take_profit_targets = [tier1_target, tier2_target, tier3_target]
            
            # Calculate risk/reward
            risk = abs(current_price - stop_loss)
            reward = abs(tier1_target - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Minimum R:R requirement
            min_rr = 1.2
            if risk_reward_ratio < min_rr:
                logger.debug(f"{symbol} R:R too low: {risk_reward_ratio:.2f} < {min_rr}")
                return None
            
            # Create signal
            signal = TradeSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                current_price=current_price,
                box_levels=box_levels,
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit_targets=take_profit_targets,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=current_time,
                confirmations=confirmation_data
            )
            
            logger.info(f"Generated signal: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
    
    def update_performance(
        self,
        symbol: str,
        trade_result: Dict[str, Any]
    ):
        """Update performance tracking after trade closes"""
        is_win = trade_result.get("pnl", 0) > 0
        pnl = trade_result.get("pnl", 0)
        entry_hour = trade_result.get("entry_time", datetime.now()).hour
        
        # Update symbol performance
        if is_win:
            self.performance_by_symbol[symbol]["wins"] += 1
        else:
            self.performance_by_symbol[symbol]["losses"] += 1
        self.performance_by_symbol[symbol]["total_pnl"] += pnl
        
        # Update hour performance
        if is_win:
            self.performance_by_hour[entry_hour]["wins"] += 1
        else:
            self.performance_by_hour[entry_hour]["losses"] += 1
        
        # Add to trade history
        self.trade_history.append(trade_result)
        
        logger.info(f"Performance updated for {symbol}: "
                   f"Win={is_win}, PnL=${pnl:.2f}, Hour={entry_hour}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }
        
        wins = [t for t in self.trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in self.trade_history if t.get("pnl", 0) <= 0]
        
        total_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
        
        return {
            "total_trades": len(self.trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trade_history) if self.trade_history else 0.0,
            "total_pnl": total_pnl,
            "avg_win": np.mean([t.get("pnl", 0) for t in wins]) if wins else 0.0,
            "avg_loss": np.mean([t.get("pnl", 0) for t in losses]) if losses else 0.0,
            "best_symbols": sorted(
                self.performance_by_symbol.items(),
                key=lambda x: x[1]["total_pnl"],
                reverse=True
            )[:5],
            "best_hours": sorted(
                [(h, s) for h, s in self.performance_by_hour.items() 
                 if s["wins"] + s["losses"] > 0],
                key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"]),
                reverse=True
            )[:3]
        }
