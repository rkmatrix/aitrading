"""
Continuous Bot Monitor - Real-time monitoring of trading bot
Monitors bot status, logs, and market conditions
"""
import time
import os
from pathlib import Path
from datetime import datetime

def get_latest_logs(log_file, lines=10):
    """Get latest log lines."""
    if not log_file.exists():
        return []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:] if line.strip()]
    except Exception:
        return []

def check_kill_switch(kill_switch_path):
    """Check kill switch status."""
    if kill_switch_path.exists():
        try:
            with open(kill_switch_path, 'r', encoding='utf-8') as f:
                content = f.read()[:200]
            return True, content
        except Exception:
            return True, "Unable to read"
    return False, None

def get_market_status():
    """Get market status."""
    try:
        from ai.market.market_clock import MarketClock
        clock = MarketClock()
        return clock.get_market_status()
    except Exception as e:
        return {"error": str(e)}

def monitor_loop(interval=30):
    """Continuous monitoring loop."""
    log_file = Path("data/logs/phase26_realtime_live.log")
    kill_switch = Path("data/runtime/trading_disabled.flag")
    
    print("=" * 70)
    print("AITradingBot Continuous Monitor")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)
    print()
    
    last_log_count = 0
    
    try:
        while True:
            # Clear screen (optional, comment out if you want to see history)
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Check")
            print("-" * 70)
            
            # Check kill switch
            kill_active, kill_content = check_kill_switch(kill_switch)
            if kill_active:
                print(f"[ALERT] KILL SWITCH ACTIVE!")
                if kill_content:
                    print(f"  Content: {kill_content[:100]}")
            else:
                print("[OK] Kill switch: Not active")
            
            # Check market status
            market_status = get_market_status()
            if "error" in market_status:
                print(f"[WARN] Market status check failed: {market_status['error']}")
            else:
                is_open = market_status.get('is_open', False)
                is_weekend = market_status.get('is_weekend', False)
                time_until_open = market_status.get('time_until_open')
                
                if is_open:
                    print(f"[ACTIVE] Market is OPEN - Bot should be trading")
                elif is_weekend:
                    print(f"[IDLE] Market is CLOSED (Weekend)")
                    if time_until_open:
                        hours = time_until_open / 3600
                        print(f"  Market opens in: {hours:.1f} hours")
                else:
                    print(f"[IDLE] Market is CLOSED")
                    if time_until_open:
                        hours = time_until_open / 3600
                        print(f"  Market opens in: {hours:.1f} hours")
            
            # Check latest logs
            logs = get_latest_logs(log_file, lines=5)
            if logs:
                current_log_count = len(logs)
                if current_log_count != last_log_count:
                    print(f"\n[LOGS] Latest activity ({len(logs)} recent entries):")
                    for log_line in logs[-3:]:  # Show last 3
                        # Clean up log line for display
                        clean_line = log_line.replace('\x00', '').replace('\ufffd', '?')
                        if len(clean_line) > 100:
                            clean_line = clean_line[:100] + "..."
                        print(f"  {clean_line}")
                    last_log_count = current_log_count
                else:
                    print(f"\n[LOGS] No new activity (last check: {len(logs)} entries)")
            else:
                print(f"\n[WARN] No log file or empty logs")
            
            print("-" * 70)
            print(f"Next check in {interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Monitoring stopped by user")
        print("=" * 70)

if __name__ == "__main__":
    import sys
    interval = 30
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print(f"Invalid interval, using default: {interval}")
    
    monitor_loop(interval=interval)
