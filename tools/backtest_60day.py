"""
Box Trading Strategy - 60-Day Backtest
---------------------------------------
Backtest using recent 60 days of 5-minute intraday data.
This provides realistic testing with actual intraday price action.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.backtest_box_trading import BoxTradingBacktester


def main():
    """Entry point for 60-day backtest"""
    print("=" * 80)
    print("BOX TRADING STRATEGY - 60-DAY BACKTEST (5-MINUTE DATA)")
    print("=" * 80)
    print()
    
    # Configuration
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
    
    # Last 60 days (5-minute data available)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Duration: 60 days (intraday 5-minute bars)")
    print()
    print("This will provide realistic intraday backtesting with:")
    print("- Actual 5-minute price bars")
    print("- Intraday support/resistance levels")
    print("- Real entry/exit timing")
    print()
    input("Press Enter to start backtesting...")
    print()
    
    # Create backtester
    backtester = BoxTradingBacktester()
    
    # Run backtest
    results = backtester.run_backtest(symbols, start_date, end_date)
    
    # Print report
    backtester.print_report()
    
    # Save results
    backtester.save_results("data/backtest_60day_results.json")
    
    print("\nBacktest complete!")
    print("Results saved to: data/backtest_60day_results.json")
    print()
    
    # Provide guidance
    if backtester.stats['total_trades'] > 0:
        print("=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print()
        
        if backtester.stats['win_rate'] >= 0.55 and backtester.stats['total_trades'] >= 20:
            print("✅ Strong performance in backtest!")
            print()
            print("Recommended actions:")
            print("1. Review the detailed results carefully")
            print("2. Understand why wins won and losses lost")
            print("3. Start paper trading to validate with live data")
            print("4. Monitor for 4+ weeks before considering live trading")
        else:
            print("⚠️  Results need improvement")
            print()
            print("Considerations:")
            if backtester.stats['total_trades'] < 20:
                print("- Too few trades executed (market may have been trending)")
                print("- Try a different time period or add more symbols")
            if backtester.stats['win_rate'] < 0.55:
                print("- Win rate below target")
                print("- Consider adjusting:")
                print("  * Zone thresholds (configs/box_trading.yaml)")
                print("  * Confirmation requirements")
                print("  * Stop loss placement")
            print()
            print("Options:")
            print("1. Adjust configuration and re-run backtest")
            print("2. Try paper trading to see live performance")
            print("3. Focus on different symbols")
    else:
        print("=" * 80)
        print("NO TRADES GENERATED")
        print("=" * 80)
        print()
        print("This period may have been strongly trending.")
        print("Box trading works best in ranging markets.")
        print()
        print("Next steps:")
        print("1. Try a different 60-day period")
        print("2. Start paper trading (will adapt to current conditions)")
        print("3. Monitor paper trading for several weeks")
    
    print()


if __name__ == "__main__":
    main()
