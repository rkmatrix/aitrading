"""Quick bot status check"""
import sys
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Bot Status Check")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check log file
log_file = Path("data/logs/phase26_realtime_live.log")
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        if lines:
            print(f"Log file: {log_file}")
            print(f"Total log entries: {len(lines)}")
            print(f"Last entry: {lines[-1].strip()[:80]}")
        else:
            print("Log file is empty")
else:
    print("Log file not found")

print()

# Check kill switch
kill_switch = Path("data/runtime/trading_disabled.flag")
if kill_switch.exists():
    print("[ALERT] KILL SWITCH IS ACTIVE!")
else:
    print("[OK] Kill switch not active")

print()

# Check market
try:
    from ai.market.market_clock import MarketClock
    clock = MarketClock()
    status = clock.get_market_status()
    print(f"Market Open: {status['is_open']}")
    print(f"Weekend: {status['is_weekend']}")
    if status.get('time_until_open'):
        hours = status['time_until_open'] / 3600
        print(f"Hours until market opens: {hours:.1f}")
except Exception as e:
    print(f"Market check error: {e}")

print()
print("=" * 60)
print("To see live logs, run:")
print("  Get-Content data\\logs\\phase26_realtime_live.log -Tail 20 -Wait")
print("=" * 60)
