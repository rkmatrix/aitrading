"""
Bot Monitor - Real-time monitoring of trading bot
"""
import time
import os
from pathlib import Path
from datetime import datetime

def monitor_bot():
    """Monitor bot status and logs."""
    log_file = Path("data/logs/phase26_realtime_live.log")
    kill_switch = Path("data/runtime/trading_disabled.flag")
    
    print("=" * 60)
    print("AITradingBot Monitor")
    print("=" * 60)
    print(f"Monitoring started at: {datetime.now()}")
    print()
    
    if log_file.exists():
        print(f"[OK] Log file found: {log_file}")
        # Read last few lines
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(f"\nLast 5 log entries:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print(f"[WARN] Log file not found: {log_file}")
    
    print()
    if kill_switch.exists():
        print(f"[ALERT] KILL SWITCH ACTIVE: {kill_switch}")
        with open(kill_switch, 'r') as f:
            content = f.read()
            print(f"   Content: {content[:100]}")
    else:
        print(f"[OK] Kill switch not active")
    
    print()
    print("Checking market status...")
    try:
        from ai.market.market_clock import MarketClock
        clock = MarketClock()
        status = clock.get_market_status()
        print(f"Market Open: {status['is_open']}")
        print(f"Regular Hours: {status['is_regular_hours']}")
        print(f"Weekend: {status['is_weekend']}")
        if status.get('time_until_open'):
            hours = status['time_until_open'] / 3600
            print(f"Time until open: {hours:.1f} hours")
    except Exception as e:
        print(f"[WARN] Could not check market status: {e}")
    
    print()
    print("=" * 60)
    print("Monitor complete. Check logs for real-time updates.")
    print("Log file: data/logs/phase26_realtime_live.log")
    print("=" * 60)

if __name__ == "__main__":
    monitor_bot()
