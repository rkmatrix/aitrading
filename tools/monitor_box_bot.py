"""
Box Trading Bot - Live Monitor
================================
Real-time monitoring of the box trading bot.
Shows current positions, today's performance, and recent activity.

Usage:
    python tools/monitor_box_bot.py              # One-time status
    python tools/monitor_box_bot.py --live        # Continuous refresh
    python tools/monitor_box_bot.py --tail 50     # Show last 50 log lines
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TRADE_JOURNAL = project_root / "data" / "box_trading_trades.json"
LOG_FILE = project_root / "data" / "logs" / "box_trading_bot.log"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_recent_log_lines(n=30):
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines[-n:]


def get_todays_trades():
    if not TRADE_JOURNAL.exists():
        return []
    with open(TRADE_JOURNAL, 'r') as f:
        journal = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")
    return [t for t in journal.get("trades", []) if t["timestamp"].startswith(today)]


def get_all_trades():
    if not TRADE_JOURNAL.exists():
        return []
    with open(TRADE_JOURNAL, 'r') as f:
        journal = json.load(f)
    return journal.get("trades", [])


def display_status():
    print("=" * 70)
    print(f"  BOX TRADING BOT - LIVE MONITOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check if bot is running (look for recent log activity)
    if LOG_FILE.exists():
        last_modified = datetime.fromtimestamp(LOG_FILE.stat().st_mtime)
        age = (datetime.now() - last_modified).total_seconds()
        if age < 120:
            print(f"  🟢 Bot Status: ACTIVE (last log {age:.0f}s ago)")
        elif age < 600:
            print(f"  🟡 Bot Status: IDLE (last log {age/60:.0f}m ago)")
        else:
            print(f"  🔴 Bot Status: INACTIVE (last log {age/3600:.1f}h ago)")
    else:
        print("  🔴 Bot Status: NO LOG FILE")

    # Market hours check
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    market_time = hour + minute/60

    if weekday >= 5:
        print("  📅 Market: CLOSED (Weekend)")
    elif 9.5 <= market_time < 16:
        print("  📈 Market: OPEN")
    elif market_time < 9.5:
        mins_to_open = int((9.5 - market_time) * 60)
        print(f"  ⏰ Market: PRE-MARKET (opens in {mins_to_open} min)")
    else:
        print("  📉 Market: CLOSED (After Hours)")

    # Today's trades
    todays = get_todays_trades()
    if todays:
        wins = sum(1 for t in todays if t["pnl"] > 0)
        losses = sum(1 for t in todays if t["pnl"] <= 0)
        total_pnl = sum(t["pnl"] for t in todays)
        wr = wins / len(todays) * 100 if todays else 0

        print(f"\n  --- TODAY'S PERFORMANCE ---")
        print(f"  Trades: {len(todays)}  |  W: {wins}  |  L: {losses}  |  WR: {wr:.0f}%")
        print(f"  P&L: ${total_pnl:,.2f}")

        print(f"\n  Recent Trades:")
        for t in todays[-5:]:
            emoji = "✅" if t["pnl"] > 0 else "❌"
            print(f"    {emoji} {t['action']} {t['symbol']} "
                  f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} "
                  f"P&L: ${t['pnl']:,.2f} ({t.get('exit_reason', 'N/A')})")
    else:
        print(f"\n  No trades today.")

    # All-time summary
    all_trades = get_all_trades()
    if all_trades:
        total = len(all_trades)
        all_wins = sum(1 for t in all_trades if t["pnl"] > 0)
        all_pnl = sum(t["pnl"] for t in all_trades)
        all_wr = all_wins / total * 100 if total else 0

        print(f"\n  --- ALL-TIME STATS ---")
        print(f"  Total Trades: {total}  |  Win Rate: {all_wr:.1f}%  |  P&L: ${all_pnl:,.2f}")

    # Recent log activity
    recent_lines = get_recent_log_lines(15)
    if recent_lines:
        print(f"\n  --- RECENT LOG ACTIVITY ---")
        for line in recent_lines:
            line = line.strip()
            if len(line) > 100:
                line = line[:97] + "..."
            if "ERROR" in line:
                print(f"  ❌ {line}")
            elif "WARNING" in line:
                print(f"  ⚠️  {line}")
            elif "signal" in line.lower() or "order" in line.lower() or "position" in line.lower():
                print(f"  📊 {line}")
            else:
                print(f"     {line}")

    print("\n" + "=" * 70)


def tail_logs(n=50):
    lines = get_recent_log_lines(n)
    print(f"\n--- Last {n} log lines ---\n")
    for line in lines:
        line = line.strip()
        if "ERROR" in line:
            print(f"❌ {line}")
        elif "WARNING" in line:
            print(f"⚠️  {line}")
        elif "signal" in line.lower() or "entry" in line.lower():
            print(f"📊 {line}")
        else:
            print(f"   {line}")
    print()


def live_monitor(interval=10):
    print("Starting live monitor (Ctrl+C to stop)...")
    try:
        while True:
            clear_screen()
            display_status()
            print(f"\n  Refreshing every {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Box Trading Bot Monitor")
    parser.add_argument("--live", action="store_true", help="Continuous live monitoring")
    parser.add_argument("--tail", type=int, default=0, help="Show last N log lines")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval for live mode")
    args = parser.parse_args()

    if args.tail > 0:
        tail_logs(args.tail)
    elif args.live:
        live_monitor(args.interval)
    else:
        display_status()


if __name__ == "__main__":
    main()
