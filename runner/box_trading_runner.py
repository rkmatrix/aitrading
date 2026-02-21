"""
Box Trading Bot Runner
----------------------
Dedicated execution loop for box trading strategy with:
- Market hours awareness (trades only during regular hours)
- Advanced risk management and circuit breakers
- Position tracking and exit management
- Telegram alerts for all trade events
- Performance monitoring and adaptive learning
"""
from __future__ import annotations

import logging
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

# Add project root to Python path
runner_dir = Path(__file__).parent
project_root = runner_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np

# Import project modules
from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify, send_trade_alert
from ai.strategies.box_trading_strategy import (
    BoxTradingStrategy,
    BoxLevels,
    TradeSignal
)
from ai.market.market_clock import MarketClock
from ai.market.enhanced_data_provider import EnhancedMarketDataProvider

# Import broker and execution components
try:
    from ai.execution.broker_alpaca_live import AlpacaLiveBroker
    BROKER_AVAILABLE = True
except ImportError as e:
    BROKER_AVAILABLE = False
    logging.warning(f"AlpacaLiveBroker not available: {e}")

# Setup logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "box_trading_bot.log"

logger = logging.getLogger("BoxTradingBot")
logger.setLevel(logging.INFO)

# File handler
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Console handler
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class Position:
    """Track active box trading position"""
    def __init__(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit_targets: List[float],
        entry_time: datetime,
        signal: TradeSignal
    ):
        self.symbol = symbol
        self.action = action  # "BUY" or "SELL"
        self.entry_price = entry_price
        self.quantity = quantity
        self.remaining_quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit_targets = take_profit_targets
        self.entry_time = entry_time
        self.signal = signal
        self.partial_exits: List[Dict[str, Any]] = []
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.action == "BUY":
            self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_quantity
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_quantity
    
    def time_in_trade(self, current_time: datetime) -> float:
        """Return minutes in trade"""
        return (current_time - self.entry_time).total_seconds() / 60
    
    def __repr__(self):
        return (f"Position({self.symbol} {self.action} {self.remaining_quantity}@{self.entry_price:.2f}, "
                f"PnL=${self.unrealized_pnl:.2f})")


class BoxTradingRunner:
    """
    Main execution loop for box trading bot
    """
    
    def __init__(self, config_path: str = "configs/box_trading.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.market_clock = MarketClock()
        self.data_provider = EnhancedMarketDataProvider()
        self.strategy = BoxTradingStrategy(self.config, self.data_provider)
        
        # Initialize broker
        self.broker = None
        self._init_broker()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.running = False
        self.start_time = None
        self.daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_pnl": 0.0
        }
        self.consecutive_losses = 0
        
        # Circuit breaker state
        self.is_paused = False
        self.pause_until = None
        self.daily_loss_triggered = False
        self._failed_order_cooldown: Dict[str, datetime] = {}
        
        logger.info("BoxTradingRunner initialized with config: %s", config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.error("Config file not found: %s", config_file)
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract box_trading section
        if 'box_trading' in config:
            config = config['box_trading']
        
        logger.info("Loaded config from %s", config_file)
        return config
    
    def _init_broker(self):
        """Initialize broker connection"""
        if not BROKER_AVAILABLE:
            logger.warning("Broker not available - running in analysis mode only")
            return
        
        try:
            ensure_env_loaded()
            
            # Get environment
            env = os.getenv("ENV", "PAPER_TRADING")
            
            if env in ["PAPER_TRADING", "LIVE"]:
                # Use the from_env() class method to create broker
                self.broker = AlpacaLiveBroker.from_env()
                logger.info("✅ Broker initialized: %s mode", env)
            else:
                logger.warning("ENV=%s not supported for box trading - use PAPER_TRADING or LIVE", env)
        
        except Exception as e:
            logger.error("Failed to initialize broker: %s", e)
            self.broker = None
    
    def _is_trading_allowed(self, current_time: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed right now
        Returns (allowed, reason)
        """
        # Check if paused
        if self.is_paused:
            if self.pause_until and current_time >= self.pause_until:
                self.is_paused = False
                self.pause_until = None
                logger.info("Pause period ended - resuming trading")
            else:
                remaining = (self.pause_until - current_time).total_seconds() / 60 if self.pause_until else 0
                return (False, f"Bot paused ({remaining:.0f} minutes remaining)")
        
        # Check circuit breakers
        circuit_config = self.config.get("circuit_breakers", {})
        
        # Daily loss limit
        if circuit_config.get("enabled", True):
            max_daily_loss = circuit_config.get("max_daily_loss_percent", 0.05)
            
            if self.daily_stats["total_pnl"] < 0:
                account_equity = self._get_account_equity()
                loss_percent = abs(self.daily_stats["total_pnl"]) / account_equity
                
                if loss_percent >= max_daily_loss:
                    if not self.daily_loss_triggered:
                        self.daily_loss_triggered = True
                        self._send_alert(
                            f"🛑 DAILY LOSS LIMIT HIT\n\n"
                            f"Loss: ${self.daily_stats['total_pnl']:.2f} ({loss_percent*100:.2f}%)\n"
                            f"Limit: {max_daily_loss*100:.0f}%\n\n"
                            f"Trading stopped for today.",
                            kind="critical"
                        )
                    return (False, "Daily loss limit exceeded")
            
            # Daily trade limit
            max_daily_trades = circuit_config.get("max_daily_trades", 15)
            if self.daily_stats["trades"] >= max_daily_trades:
                return (False, f"Daily trade limit reached ({max_daily_trades})")
            
            # Consecutive loss limit
            max_consecutive = circuit_config.get("max_consecutive_losses", 3)
            if self.consecutive_losses >= max_consecutive:
                pause_duration = circuit_config.get("pause_duration_minutes", 120)
                
                if circuit_config.get("pause_after_consecutive_losses", True):
                    self._pause_trading(pause_duration, "consecutive losses")
                    return (False, f"Paused after {max_consecutive} consecutive losses")
        
        # Check time of day
        now_et = current_time.astimezone(ZoneInfo("America/New_York"))
        current_time_only = now_et.time()
        
        # Avoid first N minutes
        avoid_first = self.config.get("avoid_first_minutes", 30)
        if avoid_first > 0:
            from datetime import time as dtime
            market_open = dtime(9, 30)
            avoid_until = dtime(9, 30 + avoid_first // 60, avoid_first % 60)
            
            if market_open <= current_time_only < avoid_until:
                return (False, f"Avoiding first {avoid_first} minutes")
        
        # Stop new trades before close
        stop_new_trades_time = self.config.get("stop_new_trades_time", "15:45")
        hour, minute = map(int, stop_new_trades_time.split(":"))
        from datetime import time as dtime
        stop_time = dtime(hour, minute)
        
        if current_time_only >= stop_time:
            return (False, "No new trades after 15:45")
        
        return (True, None)
    
    def _should_close_all_positions(self, current_time: datetime) -> bool:
        """Check if we should close all positions (end of day)"""
        close_time_str = self.config.get("close_all_positions_time", "15:55")
        hour, minute = map(int, close_time_str.split(":"))
        
        now_et = current_time.astimezone(ZoneInfo("America/New_York"))
        from datetime import time as dtime
        close_time = dtime(hour, minute)
        
        return now_et.time() >= close_time
    
    def _pause_trading(self, duration_minutes: int, reason: str):
        """Pause trading for specified duration"""
        self.is_paused = True
        self.pause_until = datetime.now(ZoneInfo("America/New_York")) + timedelta(minutes=duration_minutes)
        
        logger.warning(f"Trading paused for {duration_minutes} minutes (reason: {reason})")
        
        self._send_alert(
            f"⚠️ TRADING PAUSED\n\n"
            f"Reason: {reason}\n"
            f"Duration: {duration_minutes} minutes\n"
            f"Resume at: {self.pause_until.strftime('%H:%M:%S')}",
            kind="warning"
        )
    
    def _send_alert(self, message: str, kind: str = "info"):
        """Send Telegram alert"""
        telegram_config = self.config.get("telegram", {})
        
        if not telegram_config.get("enabled", True):
            return
        
        try:
            if kind == "critical":
                message = "🚨 " + message
            elif kind == "warning":
                message = "⚠️ " + message
            elif kind == "success":
                message = "✅ " + message
            
            notify(message, kind="orders")
        except Exception as e:
            logger.error(f"Failed to send telegram alert: {e}")
    
    def _send_trade_entry_alert(self, position: Position):
        """Send detailed trade entry alert"""
        signal = position.signal
        box = signal.box_levels
        
        message = (
            f"🎯 BOX TRADE ENTERED\n\n"
            f"Symbol: {position.symbol}\n"
            f"Action: {position.action}\n"
            f"Entry: ${position.entry_price:.2f}\n"
            f"Quantity: {position.quantity} shares\n"
            f"Stop Loss: ${position.stop_loss:.2f}\n"
            f"Targets: ${signal.take_profit_targets[0]:.2f} / "
            f"${signal.take_profit_targets[1]:.2f} / ${signal.take_profit_targets[2]:.2f}\n"
            f"Risk: ${abs(position.entry_price - position.stop_loss) * position.quantity:.2f}\n"
            f"R:R Ratio: {signal.risk_reward_ratio:.2f}:1\n\n"
            f"Box Levels:\n"
            f"- Prev High: ${box.prev_day_high:.2f}\n"
            f"- Prev Low: ${box.prev_day_low:.2f}\n"
            f"- Midpoint: ${box.midpoint:.2f}\n"
            f"- Range: {box.range_percent*100:.2f}%\n\n"
            f"Confidence: {signal.confidence*100:.0f}%\n"
            f"Reasons: {', '.join(signal.reasons)}\n"
            f"Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')} ET"
        )
        
        self._send_alert(message, kind="success")
    
    def _send_trade_exit_alert(self, position: Position, exit_price: float, exit_reason: str):
        """Send trade exit alert"""
        pnl = position.realized_pnl
        pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100 if position.quantity > 0 else 0
        
        emoji = "✅" if pnl > 0 else "❌"
        
        message = (
            f"{emoji} BOX TRADE CLOSED\n\n"
            f"Symbol: {position.symbol}\n"
            f"Action: {position.action}\n"
            f"Entry: ${position.entry_price:.2f}\n"
            f"Exit: ${exit_price:.2f}\n"
            f"Quantity: {position.quantity} shares\n"
            f"P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)\n"
            f"Duration: {position.time_in_trade(datetime.now()):.0f} minutes\n"
            f"Reason: {exit_reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
        )
        
        kind = "success" if pnl > 0 else "info"
        self._send_alert(message, kind=kind)
    
    def _check_and_execute_signals(self, current_time: datetime):
        """Check for new trade signals"""
        # Check if trading is allowed
        allowed, reason = self._is_trading_allowed(current_time)
        if not allowed:
            logger.debug(f"Trading not allowed: {reason}")
            return
        
        symbols = self.config.get("symbols", [])
        max_positions = self.config.get("max_positions", 2)
        
        # Adaptive position limit
        if self.daily_stats["trades"] >= 5:
            total = self.daily_stats["wins"] + self.daily_stats["losses"]
            win_rate = self.daily_stats["wins"] / total if total > 0 else 0
            
            if win_rate > 0.65:
                max_positions = self.config.get("max_positions_if_winning", 3)
            elif win_rate < 0.50:
                max_positions = self.config.get("max_positions_if_losing", 1)
        
        if len(self.positions) >= max_positions:
            logger.debug(f"Max positions reached: {len(self.positions)}/{max_positions}")
            return
        
        signals_found = 0
        signals_rejected = 0
        
        # Scan symbols for signals
        for symbol in symbols:
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            try:
                # Get recent data
                recent_bars = self._get_recent_bars(symbol, bars=50)
                
                if not recent_bars:
                    logger.debug(f"No recent bars for {symbol}")
                    continue
                
                current_price = recent_bars[-1].get("close", 0)
                
                if current_price <= 0:
                    continue
                
                # Generate signal
                signal = self.strategy.generate_signal(
                    symbol=symbol,
                    current_price=current_price,
                    current_time=current_time,
                    recent_bars=recent_bars
                )
                
                if signal and signal.action in ["BUY", "SELL"]:
                    signals_found += 1
                    
                    # Check confidence threshold
                    min_confidence = 0.75
                    if signal.confidence < min_confidence:
                        signals_rejected += 1
                        logger.info(f"📊 {symbol} signal rejected: confidence {signal.confidence:.2f} < {min_confidence}")
                        continue
                    
                    logger.info(f"📊 Signal detected: {signal.action} {symbol} @ ${current_price:.2f}, "
                              f"Confidence={signal.confidence:.2f}, R:R={signal.risk_reward_ratio:.2f}")
                    
                    # Execute trade
                    self._execute_entry(signal, current_time)
                    
            except Exception as e:
                logger.error(f"Error checking signal for {symbol}: {e}")
        
        if signals_found > 0:
            logger.info(f"Signal scan complete: {signals_found} signals found, {signals_rejected} rejected by confidence filter")
    
    def _get_recent_bars(self, symbol: str, bars: int = 50) -> List[Dict[str, Any]]:
        """Get recent bar data for symbol"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            hist_data = self.data_provider.get_historical_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="5m"
            )
            
            if hist_data and len(hist_data) > 0:
                return hist_data[-bars:]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent bars for {symbol}: {e}")
            return []
    
    def _wait_for_fill(self, order_id: str, timeout_seconds: int = 30) -> Optional[Dict]:
        """Wait for order to fill with timeout"""
        if not self.broker:
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                status = self.broker.get_order_status(order_id)
                if status:
                    order_status = status.get('status', '').lower()
                    if order_status in ['filled', 'partially_filled']:
                        return status
                    elif order_status in ['cancelled', 'expired', 'rejected']:
                        logger.error(f"Order {order_id} failed with status: {order_status}")
                        return None
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
            
            time.sleep(0.5)
        
        logger.warning(f"Order {order_id} fill timeout after {timeout_seconds}s")
        return None
    
    def _can_open_position_for_symbol(self, symbol: str) -> bool:
        """Check if we can open position considering correlation groups"""
        correlation_groups = self.config.get("correlation_groups", {})
        max_correlated = self.config.get("max_correlated_positions", 1)
        
        symbol_group = None
        for group_name, symbols in correlation_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break
        
        if not symbol_group:
            return True
        
        group_symbols = correlation_groups[symbol_group]
        positions_in_group = sum(1 for sym in self.positions if sym in group_symbols)
        
        return positions_in_group < max_correlated
    
    def _execute_entry(self, signal: TradeSignal, current_time: datetime):
        """Execute trade entry with real broker orders"""
        try:
            # Check cooldown from failed orders
            if signal.symbol in self._failed_order_cooldown:
                cooldown_until = self._failed_order_cooldown[signal.symbol]
                if datetime.now(ZoneInfo("America/New_York")) < cooldown_until:
                    return
                else:
                    del self._failed_order_cooldown[signal.symbol]
            
            # Check correlation groups
            if not self._can_open_position_for_symbol(signal.symbol):
                logger.info(f"Skipping {signal.symbol} - correlation group limit reached")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Position size calculation returned 0 for {signal.symbol}")
                return
            
            # Execute real broker order
            if not self.broker:
                logger.error("No broker available - cannot execute trade")
                return
            
            order = {
                "symbol": signal.symbol,
                "side": signal.action.lower(),
                "qty": position_size,
                "order_type": "market"
            }
            
            logger.info(f"Submitting {signal.action} order for {signal.symbol}: "
                       f"{position_size} shares @ ${signal.current_price:.2f}")
            
            resp = self.broker.submit_order(order)
            
            if isinstance(resp, dict):
                if resp.get("error") or resp.get("order_submitted") == False:
                    logger.error(f"Order failed for {signal.symbol}: {resp.get('error', 'Unknown error')}")
                    self._failed_order_cooldown[signal.symbol] = datetime.now(ZoneInfo("America/New_York")) + timedelta(minutes=10)
                    return
            
            order_id = resp.get('id') if isinstance(resp, dict) else (getattr(resp, 'id', None) if resp else None)
            if not order_id:
                logger.error(f"No order ID received for {signal.symbol}")
                return
            
            fill_status = self._wait_for_fill(order_id, timeout_seconds=30)
            if not fill_status:
                logger.error(f"Order {order_id} did not fill for {signal.symbol}")
                return
            
            filled_price = float(fill_status.get('filled_avg_price', signal.current_price))
            filled_qty = int(float(fill_status.get('filled_qty', position_size)))
            
            logger.info(f"✅ Order filled: {filled_qty} shares @ ${filled_price:.2f}")
            
            # Create position AFTER broker confirms
            position = Position(
                symbol=signal.symbol,
                action=signal.action,
                entry_price=filled_price,
                quantity=filled_qty,
                stop_loss=signal.stop_loss,
                take_profit_targets=signal.take_profit_targets,
                entry_time=current_time,
                signal=signal
            )
            
            self.positions[signal.symbol] = position
            
            # Update stats
            self.daily_stats["trades"] += 1
            
            # Send alert
            self._send_trade_entry_alert(position)
            
            logger.info(f"Position opened: {position}")
            
        except Exception as e:
            logger.error(f"Error executing entry for {signal.symbol}: {e}", exc_info=True)
    
    def _get_account_equity(self) -> float:
        """Get current account equity with caching"""
        if not hasattr(self, '_equity_cache'):
            self._equity_cache = {'value': 10000.0, 'timestamp': 0}
        
        current_time = time.time()
        if current_time - self._equity_cache['timestamp'] > 60:
            try:
                if self.broker and hasattr(self.broker, 'client'):
                    account = self.broker.client.get_account()
                    self._equity_cache['value'] = float(account.equity)
                    self._equity_cache['timestamp'] = current_time
                    logger.debug(f"Account equity refreshed: ${self._equity_cache['value']:.2f}")
            except Exception as e:
                logger.error(f"Failed to get account equity: {e}")
        
        return self._equity_cache['value']
    
    def _calculate_position_size(self, signal: TradeSignal) -> int:
        """Calculate position size based on risk, with buying power safety checks"""
        account_equity = self._get_account_equity()
        
        # Get risk per trade
        base_risk = self.config.get("base_risk_per_trade", 0.02)
        max_risk = self.config.get("max_risk_per_trade", 0.04)
        min_risk = self.config.get("min_risk_per_trade", 0.01)
        
        # Adjust risk based on confidence
        if signal.confidence >= 0.90:
            risk_percent = max_risk
        elif signal.confidence >= 0.80:
            risk_percent = (base_risk + max_risk) / 2
        elif signal.confidence >= 0.75:
            risk_percent = base_risk
        else:
            risk_percent = min_risk
        
        # Calculate dollar risk
        risk_amount = account_equity * risk_percent
        
        # Calculate position size based on stop distance
        stop_distance = abs(signal.current_price - signal.stop_loss)
        
        if stop_distance <= 0:
            return 0
        
        shares = int(risk_amount / stop_distance)
        
        # SAFETY: Cap position value at max % of equity (default 25%)
        max_position_pct = self.config.get("max_position_percent_of_equity", 0.25)
        max_position_value = account_equity * max_position_pct
        max_shares_by_value = int(max_position_value / signal.current_price)
        shares = min(shares, max_shares_by_value)
        
        # SAFETY: Check against buying power
        if self.broker and hasattr(self.broker, 'client'):
            try:
                account = self.broker.client.get_account()
                buying_power = float(account.buying_power)
                # Use at most 40% of available buying power per trade
                max_bp_per_trade = buying_power * 0.40
                max_shares_by_bp = int(max_bp_per_trade / signal.current_price)
                shares = min(shares, max_shares_by_bp)
                logger.debug(f"Position sizing for {signal.symbol}: risk_shares={int(risk_amount/stop_distance)}, "
                           f"value_cap={max_shares_by_value}, bp_cap={max_shares_by_bp}, final={shares}")
            except Exception as e:
                logger.warning(f"Could not check buying power: {e}")
                # Fallback: use conservative position value cap
                shares = min(shares, max_shares_by_value)
        
        # Ensure at least 1 share
        return max(1, shares)
    
    def _check_and_manage_positions(self, current_time: datetime):
        """Check and manage open positions"""
        if not self.positions:
            return
        
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            try:
                # Get current price
                current_price = self.data_provider.get_last_price(symbol)
                
                if current_price is None or current_price <= 0:
                    continue
                
                # Update P&L
                position.update_pnl(current_price)
                
                # Check stop loss
                should_stop = False
                if position.action == "BUY" and current_price <= position.stop_loss:
                    should_stop = True
                elif position.action == "SELL" and current_price >= position.stop_loss:
                    should_stop = True
                
                if should_stop:
                    logger.warning(f"Stop loss hit for {symbol} @ ${current_price:.2f}")
                    self._close_position(symbol, current_price, "Stop loss", current_time)
                    symbols_to_close.append(symbol)
                    
                    # Record stop out for whipsaw protection
                    self.strategy.record_stop_out(symbol, current_time)
                    continue
                
                # Check take profit targets
                for i, target in enumerate(position.take_profit_targets):
                    if position.action == "BUY" and current_price >= target:
                        # Hit target
                        if i == 0:  # First target - partial exit
                            self._partial_exit(position, current_price, 0.5, f"Target 1 (${target:.2f})")
                        elif i == 1:  # Second target
                            self._partial_exit(position, current_price, 0.3, f"Target 2 (${target:.2f})")
                        else:  # Final target - close
                            self._close_position(symbol, current_price, f"Target 3 (${target:.2f})", current_time)
                            symbols_to_close.append(symbol)
                            break
                    
                    elif position.action == "SELL" and current_price <= target:
                        # Hit target
                        if i == 0:  # First target - partial exit
                            self._partial_exit(position, current_price, 0.5, f"Target 1 (${target:.2f})")
                        elif i == 1:  # Second target
                            self._partial_exit(position, current_price, 0.3, f"Target 2 (${target:.2f})")
                        else:  # Final target - close
                            self._close_position(symbol, current_price, f"Target 3 (${target:.2f})", current_time)
                            symbols_to_close.append(symbol)
                            break
                
                # Check time-based exit
                max_hold_time = self.config.get("max_hold_time_minutes", 120)
                time_in_trade = position.time_in_trade(current_time)
                
                if time_in_trade >= max_hold_time and position.unrealized_pnl <= 0:
                    logger.info(f"Time-based exit for {symbol} after {time_in_trade:.0f} minutes")
                    self._close_position(symbol, current_price, "Time limit", current_time)
                    symbols_to_close.append(symbol)
                
            except Exception as e:
                logger.error(f"Error managing position for {symbol}: {e}")
        
        # Remove closed positions
        for symbol in symbols_to_close:
            if symbol in self.positions:
                del self.positions[symbol]
    
    def _partial_exit(self, position: Position, exit_price: float, percent: float, reason: str):
        """Execute partial exit"""
        exit_quantity = int(position.remaining_quantity * percent)
        
        if exit_quantity <= 0:
            return
        
        # Calculate P&L for this portion
        if position.action == "BUY":
            pnl = (exit_price - position.entry_price) * exit_quantity
        else:
            pnl = (position.entry_price - exit_price) * exit_quantity
        
        position.realized_pnl += pnl
        position.remaining_quantity -= exit_quantity
        
        position.partial_exits.append({
            "price": exit_price,
            "quantity": exit_quantity,
            "pnl": pnl,
            "reason": reason,
            "time": datetime.now()
        })
        
        logger.info(f"Partial exit for {position.symbol}: {exit_quantity} shares @ ${exit_price:.2f}, "
                   f"PnL=${pnl:.2f} ({reason})")
    
    def _close_position(self, symbol: str, exit_price: float, reason: str, current_time: datetime):
        """Close position"""
        position = self.positions.get(symbol)
        
        if not position:
            return
        
        # Calculate final P&L
        if position.action == "BUY":
            pnl = (exit_price - position.entry_price) * position.remaining_quantity
        else:
            pnl = (position.entry_price - exit_price) * position.remaining_quantity
        
        position.realized_pnl += pnl
        
        # Update stats
        is_win = position.realized_pnl > 0
        
        if is_win:
            self.daily_stats["wins"] += 1
            self.consecutive_losses = 0
        else:
            self.daily_stats["losses"] += 1
            self.consecutive_losses += 1
        
        self.daily_stats["total_pnl"] += position.realized_pnl
        
        # Update peak and drawdown
        if self.daily_stats["total_pnl"] > self.daily_stats["peak_pnl"]:
            self.daily_stats["peak_pnl"] = self.daily_stats["total_pnl"]
        
        drawdown = self.daily_stats["peak_pnl"] - self.daily_stats["total_pnl"]
        if drawdown > self.daily_stats["max_drawdown"]:
            self.daily_stats["max_drawdown"] = drawdown
        
        # Update strategy performance
        trade_result = {
            "symbol": symbol,
            "action": position.action,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": position.realized_pnl,
            "entry_time": position.entry_time,
            "exit_time": current_time,
            "duration_minutes": position.time_in_trade(current_time),
            "reason": reason
        }
        
        self.strategy.update_performance(symbol, trade_result)
        
        # Send alert
        self._send_trade_exit_alert(position, exit_price, reason)
        
        # Record to trade journal
        self._record_trade(position, exit_price, reason, position.realized_pnl)
        
        logger.info(f"Position closed: {symbol} @ ${exit_price:.2f}, PnL=${position.realized_pnl:.2f} ({reason})")
    
    def _record_trade(self, position: Position, exit_price: float, exit_reason: str, total_pnl: float):
        """Record completed trade to journal file"""
        try:
            journal_path = Path("data/box_trading_trades.json")
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            
            trade_record = {
                "trade_id": f"BOX_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{position.symbol}",
                "timestamp": datetime.now().isoformat(),
                "symbol": position.symbol,
                "action": position.action,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time.isoformat(),
                "exit_price": exit_price,
                "exit_time": datetime.now().isoformat(),
                "quantity": position.quantity,
                "remaining_quantity": position.remaining_quantity,
                "stop_loss": position.stop_loss,
                "take_profit_targets": position.take_profit_targets,
                "exit_reason": exit_reason,
                "pnl": total_pnl,
                "pnl_percent": (total_pnl / (position.entry_price * position.quantity)) * 100 if position.quantity > 0 else 0,
                "hold_time_minutes": position.time_in_trade(datetime.now()),
                "partial_exits": len(position.partial_exits),
                "confidence": position.signal.confidence if position.signal else 0,
                "risk_reward": position.signal.risk_reward_ratio if position.signal else 0,
                "box_levels": {
                    "prev_high": position.signal.box_levels.prev_day_high,
                    "prev_low": position.signal.box_levels.prev_day_low,
                    "midpoint": position.signal.box_levels.midpoint
                } if position.signal and position.signal.box_levels else {}
            }
            
            # Load existing journal
            if journal_path.exists():
                with open(journal_path, 'r') as f:
                    journal = json.load(f)
            else:
                journal = {"trades": [], "summary": {}}
            
            journal["trades"].append(trade_record)
            
            # Update summary
            trades = journal["trades"]
            total_trades = len(trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = sum(1 for t in trades if t["pnl"] <= 0)
            total_pnl_all = sum(t["pnl"] for t in trades)
            
            journal["summary"] = {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total_trades if total_trades > 0 else 0,
                "total_pnl": total_pnl_all,
                "avg_pnl": total_pnl_all / total_trades if total_trades > 0 else 0,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(journal_path, 'w') as f:
                json.dump(journal, f, indent=2, default=str)
            
            logger.info(f"Trade recorded: {trade_record['trade_id']}, PnL=${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _close_all_positions(self, reason: str, current_time: datetime):
        """Close all open positions"""
        if not self.positions:
            return
        
        logger.info(f"Closing all positions: {reason}")
        
        successfully_closed = []
        for symbol in list(self.positions.keys()):
            try:
                current_price = self.data_provider.get_last_price(symbol)
                
                if current_price and current_price > 0:
                    self._close_position(symbol, current_price, reason, current_time)
                    successfully_closed.append(symbol)
                else:
                    logger.error(f"No price available for {symbol} - cannot close")
                    
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
        
        # Only remove successfully closed positions
        for symbol in successfully_closed:
            if symbol in self.positions:
                del self.positions[symbol]
        
        remaining = len(self.positions)
        if remaining > 0:
            logger.warning(f"⚠️ {remaining} positions could not be closed: {list(self.positions.keys())}")
    
    def _send_daily_summary(self):
        """Send end of day summary"""
        stats = self.strategy.get_performance_stats()
        
        win_rate = (self.daily_stats["wins"] / self.daily_stats["trades"]) if self.daily_stats["trades"] > 0 else 0
        
        message = (
            f"📊 BOX TRADING - DAILY SUMMARY\n\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"Trades: {self.daily_stats['trades']}\n"
            f"Wins: {self.daily_stats['wins']}\n"
            f"Losses: {self.daily_stats['losses']}\n"
            f"Win Rate: {win_rate*100:.1f}%\n\n"
            f"Total P&L: ${self.daily_stats['total_pnl']:.2f}\n"
            f"Max Drawdown: ${self.daily_stats['max_drawdown']:.2f}\n\n"
            f"Overall Stats (All Time):\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate']*100:.1f}%\n"
            f"Total P&L: ${stats['total_pnl']:.2f}\n"
            f"Avg Win: ${stats['avg_win']:.2f}\n"
            f"Avg Loss: ${stats['avg_loss']:.2f}"
        )
        
        self._send_alert(message, kind="info")
    
    def _send_validation_reminder(self):
        """Send reminder to validate after 4 weeks of paper trading"""
        stats = self.strategy.get_performance_stats()
        
        # Calculate runtime
        runtime_days = (datetime.now() - self.start_time).days
        
        message = (
            f"🎯 VALIDATION REMINDER - 4 WEEKS COMPLETED!\n\n"
            f"Your Box Trading Bot has been running for {runtime_days} days.\n"
            f"Time to validate performance before going LIVE!\n\n"
            f"Current Stats:\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate']*100:.1f}%\n"
            f"Total P&L: ${stats['total_pnl']:.2f}\n\n"
            f"📋 VALIDATION REQUIREMENTS:\n"
            f"✓ Min 50 trades (you have {stats['total_trades']})\n"
            f"✓ Win rate >55% (you have {stats['win_rate']*100:.1f}%)\n"
            f"✓ Profit factor >1.4\n"
            f"✓ Max drawdown <8%\n\n"
            f"📝 NEXT STEPS:\n"
            f"1. Run validation tool:\n"
            f"   python tools\\validate_box_trading.py\n\n"
            f"2. Review detailed results\n\n"
            f"3. If PASSED: Consider Live Phase 1\n"
            f"   - Change ENV=LIVE in .env\n"
            f"   - Start with $100 max position\n"
            f"   - Update current_phase in config\n\n"
            f"4. If FAILED: Continue paper trading\n"
            f"   - Analyze losing trades\n"
            f"   - Adjust configuration\n"
            f"   - Re-test for another 2-4 weeks\n\n"
            f"⚠️ DO NOT go live without validation!"
        )
        
        logger.info("Sending 4-week validation reminder")
        self._send_alert(message, kind="info")
    
    
    def _idle(self, reason: str, sleep_seconds: float = 60.0):
        """Idle when market closed or conditions not met"""
        logger.debug(f"Idling: {reason}")
        time.sleep(sleep_seconds)
    
    def run(self):
        """Main execution loop"""
        self.running = True
        self.start_time = datetime.now(ZoneInfo("America/New_York"))
        
        # Track 4-week validation reminder
        self.validation_reminder_sent = False
        self.four_weeks_date = self.start_time + timedelta(weeks=4)
        
        logger.info("=" * 60)
        logger.info("Box Trading Bot Starting")
        logger.info("=" * 60)
        logger.info("Config: %s", self.config_path)
        logger.info("Symbols: %s", self.config.get("symbols", []))
        logger.info("Max Positions: %s", self.config.get("max_positions", 2))
        logger.info("=" * 60)
        
        # Send startup alert
        self._send_alert(
            f"🚀 BOX TRADING BOT STARTED\n\n"
            f"Symbols: {', '.join(self.config.get('symbols', []))}\n"
            f"Max Positions: {self.config.get('max_positions', 2)}\n"
            f"Risk per Trade: {self.config.get('base_risk_per_trade', 0.02)*100:.0f}%\n\n"
            f"📅 Validation reminder set for: {self.four_weeks_date.strftime('%Y-%m-%d')}\n"
            f"(After 4 weeks of paper trading)",
            kind="success"
        )
        
        last_daily_reset = datetime.now().date()
        last_summary_sent = False
        
        try:
            while self.running:
                current_time = datetime.now(ZoneInfo("America/New_York"))
                
                # Check if new day - reset daily stats
                if current_time.date() != last_daily_reset:
                    logger.info("New trading day - resetting daily stats")
                    self.daily_stats = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_pnl": 0.0,
                        "max_drawdown": 0.0,
                        "peak_pnl": 0.0
                    }
                    self.daily_loss_triggered = False
                    last_daily_reset = current_time.date()
                    last_summary_sent = False
                    
                    # Check if 4 weeks have passed (validation reminder)
                    if not self.validation_reminder_sent and current_time >= self.four_weeks_date:
                        self._send_validation_reminder()
                        self.validation_reminder_sent = True
                
                # Check market hours
                if not self.market_clock.is_regular_hours():
                    status = self.market_clock.get_market_status()
                    
                    # Send daily summary if not sent yet
                    if not last_summary_sent and self.daily_stats["trades"] > 0:
                        if status.get("is_weekend") or current_time.hour >= 17:
                            self._send_daily_summary()
                            last_summary_sent = True
                    
                    self._idle("Market closed", sleep_seconds=60.0)
                    continue
                
                # Check if we should close all positions (end of day)
                if self._should_close_all_positions(current_time):
                    self._close_all_positions("End of day", current_time)
                    
                    # Send daily summary
                    if not last_summary_sent:
                        self._send_daily_summary()
                        last_summary_sent = True
                    
                    self._idle("After hours", sleep_seconds=60.0)
                    continue
                
                # Manage existing positions
                self._check_and_manage_positions(current_time)
                
                # Check for new signals
                self._check_and_execute_signals(current_time)
                
                # Sleep between iterations
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - shutting down")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self._send_alert(
                f"❌ BOT ERROR\n\n{str(e)[:200]}",
                kind="critical"
            )
        finally:
            self.running = False
            
            # Close all positions on shutdown
            if self.positions:
                logger.info("Closing all positions before shutdown")
                self._close_all_positions("Bot shutdown", datetime.now(ZoneInfo("America/New_York")))
            
            logger.info("Box Trading Bot stopped")
            
            self._send_alert(
                f"🛑 BOX TRADING BOT STOPPED\n\n"
                f"Runtime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours",
                kind="info"
            )


def main():
    """Entry point"""
    # Load environment
    ensure_env_loaded()
    
    # Check environment
    env = os.getenv("ENV", "PAPER_TRADING")
    
    if env not in ["PAPER_TRADING", "LIVE"]:
        logger.error(f"Invalid ENV: {env}. Use PAPER_TRADING or LIVE for box trading.")
        sys.exit(1)
    
    logger.info(f"Starting box trading bot in {env} mode")
    
    # Create and run bot
    bot = BoxTradingRunner(config_path="configs/box_trading.yaml")
    bot.run()


if __name__ == "__main__":
    main()
