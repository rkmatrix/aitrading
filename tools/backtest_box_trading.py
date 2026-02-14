"""
Box Trading Strategy Backtester
--------------------------------
Backtest the box trading strategy using historical data.
Tests strategy performance over multiple years to validate approach.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd

from ai.strategies.box_trading_strategy import BoxTradingStrategy, BoxLevels
from ai.market.enhanced_data_provider import EnhancedMarketDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BoxTradingBacktest")


class BacktestPosition:
    """Simulated position for backtesting"""
    def __init__(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit_targets: List[float],
        entry_time: datetime
    ):
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.quantity = quantity
        self.remaining_quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit_targets = take_profit_targets
        self.entry_time = entry_time
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.exit_reason = None
        self.bars_held = 0
        
    def update(self, current_price: float, current_time: datetime, bar_num: int):
        """Update position with current price"""
        self.bars_held = bar_num
        
        # Calculate unrealized PnL
        if self.action == "BUY":
            unrealized = (current_price - self.entry_price) * self.remaining_quantity
        else:
            unrealized = (self.entry_price - current_price) * self.remaining_quantity
        
        return unrealized
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position"""
        if self.action == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.realized_pnl = (exit_price - self.entry_price) * self.remaining_quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.realized_pnl = (self.entry_price - exit_price) * self.remaining_quantity
        
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason


class BoxTradingBacktester:
    """
    Backtests box trading strategy on historical data
    """
    
    def __init__(self, config_path: str = "configs/box_trading.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_provider = EnhancedMarketDataProvider()
        self.strategy = BoxTradingStrategy(self.config, self.data_provider)
        
        # Backtest state
        self.positions: List[BacktestPosition] = []
        self.closed_positions: List[BacktestPosition] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.starting_capital = 10000.0  # $10k starting capital
        self.current_capital = self.starting_capital
        self.peak_capital = self.starting_capital
        
        # Statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_bars_held": 0.0,
            "trades_by_symbol": defaultdict(int),
            "pnl_by_symbol": defaultdict(float),
            "trades_by_year": defaultdict(int),
            "pnl_by_year": defaultdict(float)
        }
        
        logger.info("Backtester initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'box_trading' in config:
            config = config['box_trading']
        
        return config
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get historical data for backtesting
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical data for {symbol}: {start_date} to {end_date}")
        
        try:
            import yfinance as yf
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            
            if df.empty:
                # Try daily data if 5m not available
                logger.warning(f"5-minute data not available for {symbol}, using daily")
                df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                logger.error(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            logger.info(f"Loaded {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, signal_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        risk_per_trade = self.config.get("base_risk_per_trade", 0.02)
        risk_amount = self.current_capital * risk_per_trade
        
        stop_distance = abs(signal_price - stop_loss)
        if stop_distance <= 0:
            return 0
        
        shares = int(risk_amount / stop_distance)
        return max(1, shares)
    
    def check_exit_conditions(
        self,
        position: BacktestPosition,
        current_bar: pd.Series,
        bar_num: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited
        Returns (should_exit, reason)
        """
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        # Check stop loss
        if position.action == "BUY":
            if current_low <= position.stop_loss:
                return (True, "Stop Loss")
        else:  # SELL
            if current_high >= position.stop_loss:
                return (True, "Stop Loss")
        
        # Check take profit targets
        if position.action == "BUY":
            if current_high >= position.take_profit_targets[0]:
                return (True, "Target 1")
        else:  # SELL
            if current_low <= position.take_profit_targets[0]:
                return (True, "Target 1")
        
        # Check time-based exit (max hold time)
        max_bars = self.config.get("max_hold_time_minutes", 120) // 5  # Convert to 5-min bars
        if bar_num >= max_bars:
            # Only exit if not profitable
            unrealized = position.update(current_price, current_bar.name, bar_num)
            if unrealized <= 0:
                return (True, "Time Limit")
        
        return (False, None)
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            symbols: List of symbols to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Starting capital: ${self.starting_capital:,.2f}")
        
        # Track data for all symbols
        symbol_data = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date)
            if not df.empty:
                symbol_data[symbol] = df
        
        if not symbol_data:
            logger.error("No data loaded for any symbols")
            return {}
        
        # Get all unique dates (daily level)
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df.index.date)
        
        all_dates = sorted(all_dates)
        logger.info(f"Processing {len(all_dates)} trading days")
        
        # Process each day
        for day_idx, date in enumerate(all_dates):
            # Update equity curve
            self.equity_curve.append((datetime.combine(date, datetime.min.time()), self.current_capital))
            
            # Update peak capital and drawdown
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            drawdown = self.peak_capital - self.current_capital
            drawdown_pct = drawdown / self.peak_capital if self.peak_capital > 0 else 0
            
            if drawdown > self.stats["max_drawdown"]:
                self.stats["max_drawdown"] = drawdown
                self.stats["max_drawdown_pct"] = drawdown_pct
            
            # Check each symbol
            for symbol, df in symbol_data.items():
                # Get today's bars
                day_bars = df[df.index.date == date]
                
                if day_bars.empty:
                    continue
                
                # Calculate box levels from previous day
                # Get previous day's data
                prev_date_idx = all_dates.index(date) - 1
                if prev_date_idx < 0:
                    continue
                
                prev_date = all_dates[prev_date_idx]
                prev_day_bars = df[df.index.date == prev_date]
                
                if prev_day_bars.empty:
                    continue
                
                prev_day_high = prev_day_bars['high'].max()
                prev_day_low = prev_day_bars['low'].min()
                prev_day_close = prev_day_bars['close'].iloc[-1]
                prev_day_volume = prev_day_bars['volume'].sum()
                
                # Create box levels
                box_levels = BoxLevels(
                    symbol=symbol,
                    prev_day_high=prev_day_high,
                    prev_day_low=prev_day_low,
                    prev_day_close=prev_day_close,
                    prev_day_volume=prev_day_volume,
                    timestamp=datetime.combine(date, datetime.min.time())
                )
                
                # Process each bar of the day
                for bar_idx, (timestamp, bar) in enumerate(day_bars.iterrows()):
                    # Skip first 30 minutes
                    avoid_first_mins = self.config.get("avoid_first_minutes", 30)
                    if bar_idx < avoid_first_mins // 5:  # 5-min bars
                        continue
                    
                    # Stop new trades after 3:45 PM
                    if timestamp.hour >= 15 and timestamp.minute >= 45:
                        break
                    
                    # Check existing positions for exits
                    for position in self.positions[:]:
                        if position.symbol != symbol:
                            continue
                        
                        should_exit, reason = self.check_exit_conditions(position, bar, bar_idx)
                        
                        if should_exit:
                            # Close position
                            exit_price = bar['close']
                            position.close(exit_price, timestamp, reason)
                            
                            # Update capital
                            self.current_capital += position.realized_pnl
                            
                            # Track statistics
                            self.stats["total_trades"] += 1
                            
                            if position.realized_pnl > 0:
                                self.stats["winning_trades"] += 1
                            else:
                                self.stats["losing_trades"] += 1
                            
                            self.stats["total_pnl"] += position.realized_pnl
                            self.stats["trades_by_symbol"][symbol] += 1
                            self.stats["pnl_by_symbol"][symbol] += position.realized_pnl
                            self.stats["trades_by_year"][timestamp.year] += 1
                            self.stats["pnl_by_year"][timestamp.year] += position.realized_pnl
                            
                            if position.realized_pnl > self.stats["best_trade"]:
                                self.stats["best_trade"] = position.realized_pnl
                            if position.realized_pnl < self.stats["worst_trade"]:
                                self.stats["worst_trade"] = position.realized_pnl
                            
                            # Move to closed positions
                            self.closed_positions.append(position)
                            self.positions.remove(position)
                            
                            logger.debug(f"Closed {symbol} {position.action} @ ${exit_price:.2f}, "
                                       f"PnL: ${position.realized_pnl:.2f} ({reason})")
                    
                    # Check for new signals (if we have capacity)
                    max_positions = self.config.get("max_positions", 2)
                    if len(self.positions) >= max_positions:
                        continue
                    
                    # Check if already have position in this symbol
                    if any(p.symbol == symbol for p in self.positions):
                        continue
                    
                    # Convert day_bars to list of dicts for strategy
                    recent_bars = []
                    for _, row in day_bars.iloc[:bar_idx+1].iterrows():
                        recent_bars.append({
                            "open": row['open'],
                            "high": row['high'],
                            "low": row['low'],
                            "close": row['close'],
                            "volume": row['volume']
                        })
                    
                    if len(recent_bars) < 20:
                        continue
                    
                    # Generate signal
                    signal = self.strategy.generate_signal(
                        symbol=symbol,
                        current_price=bar['close'],
                        current_time=timestamp,
                        recent_bars=recent_bars
                    )
                    
                    if signal and signal.action in ["BUY", "SELL"]:
                        # Calculate position size
                        quantity = self.calculate_position_size(signal.current_price, signal.stop_loss)
                        
                        if quantity <= 0:
                            continue
                        
                        # Check if we have enough capital
                        position_value = signal.current_price * quantity
                        if position_value > self.current_capital * 0.5:  # Max 50% per trade
                            quantity = int((self.current_capital * 0.5) / signal.current_price)
                        
                        if quantity <= 0:
                            continue
                        
                        # Create position
                        position = BacktestPosition(
                            symbol=symbol,
                            action=signal.action,
                            entry_price=signal.current_price,
                            quantity=quantity,
                            stop_loss=signal.stop_loss,
                            take_profit_targets=signal.take_profit_targets,
                            entry_time=timestamp
                        )
                        
                        self.positions.append(position)
                        
                        logger.debug(f"Opened {symbol} {signal.action} @ ${signal.current_price:.2f}, "
                                   f"Qty: {quantity}, Confidence: {signal.confidence:.2f}")
        
        # Close any remaining positions at end
        if self.positions:
            logger.info(f"Closing {len(self.positions)} remaining positions")
            for position in self.positions:
                # Get last bar for this symbol
                last_bar = symbol_data[position.symbol].iloc[-1]
                position.close(last_bar['close'], last_bar.name, "End of Backtest")
                self.current_capital += position.realized_pnl
                self.closed_positions.append(position)
        
        # Calculate final statistics
        self._calculate_final_stats()
        
        logger.info("Backtest complete!")
        
        return self.stats
    
    def _calculate_final_stats(self):
        """Calculate final statistics"""
        # Always set final capital
        self.stats["final_capital"] = self.current_capital
        self.stats["total_return"] = (self.current_capital - self.starting_capital) / self.starting_capital
        self.stats["total_return_pct"] = self.stats["total_return"] * 100
        
        if not self.closed_positions:
            return
        
        wins = [p for p in self.closed_positions if p.realized_pnl > 0]
        losses = [p for p in self.closed_positions if p.realized_pnl <= 0]
        
        # Win rate
        self.stats["win_rate"] = len(wins) / len(self.closed_positions) if self.closed_positions else 0
        
        # Average win/loss
        self.stats["avg_win"] = np.mean([p.realized_pnl for p in wins]) if wins else 0
        self.stats["avg_loss"] = np.mean([p.realized_pnl for p in losses]) if losses else 0
        
        # Profit factor
        total_wins = sum(p.realized_pnl for p in wins)
        total_losses = abs(sum(p.realized_pnl for p in losses))
        self.stats["profit_factor"] = total_wins / total_losses if total_losses > 0 else 0
        
        # Average bars held
        self.stats["avg_bars_held"] = np.mean([p.bars_held for p in self.closed_positions])
        
        # Sharpe ratio
        returns = [p.realized_pnl / self.starting_capital for p in self.closed_positions]
        if len(returns) > 1:
            self.stats["sharpe_ratio"] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for p in self.closed_positions:
            if p.realized_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        self.stats["max_consecutive_wins"] = max_consec_wins
        self.stats["max_consecutive_losses"] = max_consec_losses
        
        # Final equity
        self.stats["final_capital"] = self.current_capital
        self.stats["total_return"] = (self.current_capital - self.starting_capital) / self.starting_capital
        self.stats["total_return_pct"] = self.stats["total_return"] * 100
    
    def print_report(self):
        """Print backtest report"""
        print("\n" + "=" * 80)
        print("BOX TRADING STRATEGY - BACKTEST RESULTS")
        print("=" * 80)
        print()
        
        # Check if any trades were executed
        if self.stats['total_trades'] == 0:
            print("⚠️  NO TRADES EXECUTED")
            print()
            print("Possible reasons:")
            print("- Using daily data (strategy designed for intraday 5-minute bars)")
            print("- No valid signals generated (all confirmations required)")
            print("- Market conditions didn't match strategy requirements")
            print()
            print("Recommendation:")
            print("1. Test with recent 60-day period (5-minute data available)")
            print("2. Or run paper trading to see live performance")
            print()
            print("=" * 80)
            return
        
        print(f"Starting Capital:      ${self.starting_capital:,.2f}")
        final_capital = self.stats.get('final_capital', self.current_capital)
        print(f"Final Capital:         ${final_capital:,.2f}")
        print(f"Total Return:          {self.stats['total_return_pct']:.2f}%")
        print(f"Total P&L:             ${self.stats['total_pnl']:,.2f}")
        print()
        
        print("TRADE STATISTICS:")
        print("-" * 80)
        print(f"Total Trades:          {self.stats['total_trades']}")
        print(f"Winning Trades:        {self.stats['winning_trades']} ({self.stats['win_rate']*100:.1f}%)")
        print(f"Losing Trades:         {self.stats['losing_trades']}")
        print()
        
        print(f"Average Win:           ${self.stats['avg_win']:.2f}")
        print(f"Average Loss:          ${self.stats['avg_loss']:.2f}")
        print(f"Best Trade:            ${self.stats['best_trade']:.2f}")
        print(f"Worst Trade:           ${self.stats['worst_trade']:.2f}")
        print(f"Profit Factor:         {self.stats['profit_factor']:.2f}")
        print()
        
        print(f"Max Drawdown:          ${self.stats['max_drawdown']:.2f} ({self.stats['max_drawdown_pct']*100:.1f}%)")
        print(f"Sharpe Ratio:          {self.stats['sharpe_ratio']:.2f}")
        print(f"Avg Hold Time:         {self.stats['avg_bars_held']*5:.0f} minutes")
        print(f"Max Consecutive Wins:  {self.stats['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {self.stats['max_consecutive_losses']}")
        print()
        
        print("PERFORMANCE BY SYMBOL:")
        print("-" * 80)
        for symbol in sorted(self.stats['trades_by_symbol'].keys()):
            trades = self.stats['trades_by_symbol'][symbol]
            pnl = self.stats['pnl_by_symbol'][symbol]
            print(f"{symbol:6s}  Trades: {trades:4d}  P&L: ${pnl:>10,.2f}")
        print()
        
        print("PERFORMANCE BY YEAR:")
        print("-" * 80)
        for year in sorted(self.stats['trades_by_year'].keys()):
            trades = self.stats['trades_by_year'][year]
            pnl = self.stats['pnl_by_year'][year]
            print(f"{year}    Trades: {trades:4d}  P&L: ${pnl:>10,.2f}")
        print()
        
        print("=" * 80)
        
        # Validation against requirements
        print("\nVALIDATION AGAINST REQUIREMENTS:")
        print("-" * 80)
        
        requirements = self.config.get("paper_trading_requirements", {})
        
        min_trades = requirements.get("min_trades", 50)
        passed_trades = self.stats['total_trades'] >= min_trades
        print(f"{'✅' if passed_trades else '❌'} Min Trades: {self.stats['total_trades']} / {min_trades} required")
        
        min_win_rate = requirements.get("min_win_rate", 0.55)
        passed_win_rate = self.stats['win_rate'] >= min_win_rate
        print(f"{'✅' if passed_win_rate else '❌'} Win Rate: {self.stats['win_rate']*100:.1f}% / {min_win_rate*100:.0f}% required")
        
        min_pf = requirements.get("min_profit_factor", 1.4)
        passed_pf = self.stats['profit_factor'] >= min_pf
        print(f"{'✅' if passed_pf else '❌'} Profit Factor: {self.stats['profit_factor']:.2f} / {min_pf} required")
        
        max_dd = requirements.get("max_drawdown_percent", 0.08)
        passed_dd = self.stats['max_drawdown_pct'] <= max_dd
        print(f"{'✅' if passed_dd else '❌'} Max Drawdown: {self.stats['max_drawdown_pct']*100:.1f}% / {max_dd*100:.0f}% limit")
        
        all_passed = passed_trades and passed_win_rate and passed_pf and passed_dd
        
        print()
        if all_passed:
            print("✅ STRATEGY VALIDATED - Ready for paper trading!")
        else:
            print("❌ STRATEGY NEEDS IMPROVEMENT - Review parameters")
        print()
    
    def save_results(self, filename: str = "data/backtest_results.json"):
        """Save backtest results to file"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "config": self.config_path,
            "starting_capital": self.starting_capital,
            "stats": dict(self.stats),
            "trades": [
                {
                    "symbol": p.symbol,
                    "action": p.action,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "quantity": p.quantity,
                    "pnl": p.realized_pnl,
                    "entry_time": p.entry_time.isoformat() if p.entry_time else None,
                    "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                    "exit_reason": p.exit_reason,
                    "bars_held": p.bars_held
                }
                for p in self.closed_positions
            ],
            "equity_curve": [
                {"time": t.isoformat(), "equity": e}
                for t, e in self.equity_curve
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Entry point"""
    print("=" * 80)
    print("BOX TRADING STRATEGY - HISTORICAL BACKTEST")
    print("=" * 80)
    print()
    
    # Configuration
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
    
    # 5 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Duration: 5 years")
    print()
    print("Note: Downloading 5 years of 5-minute data may take several minutes...")
    print("If 5-minute data unavailable, will use daily data instead.")
    print()
    input("Press Enter to continue...")
    print()
    
    # Create backtester
    backtester = BoxTradingBacktester()
    
    # Run backtest
    results = backtester.run_backtest(symbols, start_date, end_date)
    
    # Print report
    backtester.print_report()
    
    # Save results
    backtester.save_results()
    
    print("\nBacktest complete!")
    print("Results saved to: data/backtest_results.json")
    print()


if __name__ == "__main__":
    main()
