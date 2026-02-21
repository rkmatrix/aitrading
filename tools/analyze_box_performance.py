"""
Box Trading Bot - Performance Analyzer
========================================
Analyzes trade journal and logs to provide actionable insights.

Usage:
    python tools/analyze_box_performance.py
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TRADE_JOURNAL = project_root / "data" / "box_trading_trades.json"
LOG_FILE = project_root / "data" / "logs" / "box_trading_bot.log"


def load_journal():
    if not TRADE_JOURNAL.exists():
        print("No trade journal found at:", TRADE_JOURNAL)
        print("The bot needs to execute trades first to create the journal.")
        return None
    with open(TRADE_JOURNAL, 'r') as f:
        return json.load(f)


def analyze_trades(journal):
    trades = journal.get("trades", [])
    if not trades:
        print("No trades recorded yet.")
        return

    print("=" * 70)
    print("  BOX TRADING BOT - PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"  Total Trades: {len(trades)}")
    print(f"  Period: {trades[0]['timestamp'][:10]} to {trades[-1]['timestamp'][:10]}")
    print("=" * 70)

    # Basic stats
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    print(f"\n--- OVERALL PERFORMANCE ---")
    print(f"  Wins: {len(wins)}  |  Losses: {len(losses)}  |  Win Rate: {win_rate:.1f}%")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Avg P&L per Trade: ${total_pnl / len(trades):,.2f}")

    if wins:
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        max_win = max(t["pnl"] for t in wins)
        print(f"  Avg Win: ${avg_win:,.2f}  |  Best Win: ${max_win:,.2f}")

    if losses:
        avg_loss = sum(t["pnl"] for t in losses) / len(losses)
        max_loss = min(t["pnl"] for t in losses)
        print(f"  Avg Loss: ${avg_loss:,.2f}  |  Worst Loss: ${max_loss:,.2f}")

    if wins and losses:
        profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses))
        print(f"  Profit Factor: {profit_factor:.2f}")

    # By symbol
    print(f"\n--- BY SYMBOL ---")
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)

    print(f"  {'Symbol':<8} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*12} {'-'*10}")
    for sym in sorted(by_symbol.keys()):
        sym_trades = by_symbol[sym]
        sym_wins = sum(1 for t in sym_trades if t["pnl"] > 0)
        sym_wr = sym_wins / len(sym_trades) * 100 if sym_trades else 0
        sym_pnl = sum(t["pnl"] for t in sym_trades)
        sym_avg = sym_pnl / len(sym_trades) if sym_trades else 0
        print(f"  {sym:<8} {len(sym_trades):>7} {sym_wins:>6} {sym_wr:>7.1f}% ${sym_pnl:>10,.2f} ${sym_avg:>8,.2f}")

    # By hour
    print(f"\n--- BY HOUR (Entry Time) ---")
    by_hour = defaultdict(list)
    for t in trades:
        try:
            entry_hour = datetime.fromisoformat(t["entry_time"]).hour
            by_hour[entry_hour].append(t)
        except (ValueError, KeyError):
            pass

    if by_hour:
        print(f"  {'Hour':>6} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>12}")
        print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*12}")
        for hour in sorted(by_hour.keys()):
            h_trades = by_hour[hour]
            h_wins = sum(1 for t in h_trades if t["pnl"] > 0)
            h_wr = h_wins / len(h_trades) * 100 if h_trades else 0
            h_pnl = sum(t["pnl"] for t in h_trades)
            print(f"  {hour:>4}:00 {len(h_trades):>7} {h_wr:>7.1f}% ${h_pnl:>10,.2f}")

    # By action (BUY vs SELL)
    print(f"\n--- BY ACTION ---")
    by_action = defaultdict(list)
    for t in trades:
        by_action[t["action"]].append(t)

    for action in sorted(by_action.keys()):
        a_trades = by_action[action]
        a_wins = sum(1 for t in a_trades if t["pnl"] > 0)
        a_wr = a_wins / len(a_trades) * 100 if a_trades else 0
        a_pnl = sum(t["pnl"] for t in a_trades)
        print(f"  {action}: {len(a_trades)} trades, {a_wr:.1f}% win rate, ${a_pnl:,.2f} P&L")

    # By exit reason
    print(f"\n--- BY EXIT REASON ---")
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t.get("exit_reason", "Unknown")].append(t)

    for reason in sorted(by_reason.keys()):
        r_trades = by_reason[reason]
        r_pnl = sum(t["pnl"] for t in r_trades)
        print(f"  {reason}: {len(r_trades)} trades, ${r_pnl:,.2f} P&L")

    # Hold time analysis
    print(f"\n--- HOLD TIME ANALYSIS ---")
    hold_times = [t.get("hold_time_minutes", 0) for t in trades if t.get("hold_time_minutes")]
    if hold_times:
        win_hold = [t.get("hold_time_minutes", 0) for t in wins if t.get("hold_time_minutes")]
        loss_hold = [t.get("hold_time_minutes", 0) for t in losses if t.get("hold_time_minutes")]
        print(f"  Avg Hold Time: {sum(hold_times)/len(hold_times):.1f} min")
        if win_hold:
            print(f"  Avg Win Hold: {sum(win_hold)/len(win_hold):.1f} min")
        if loss_hold:
            print(f"  Avg Loss Hold: {sum(loss_hold)/len(loss_hold):.1f} min")

    # Streak analysis
    print(f"\n--- STREAK ANALYSIS ---")
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    streak_type = None
    for t in trades:
        if t["pnl"] > 0:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    print(f"  Max Win Streak: {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")

    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    if win_rate < 50:
        print("  ⚠️  Win rate below 50% - consider tightening entry filters")
    elif win_rate < 55:
        print("  ⚠️  Win rate below target 55% - monitor closely")
    else:
        print("  ✅ Win rate on target")

    if len(trades) < 50:
        remaining = 50 - len(trades)
        print(f"  📊 Need {remaining} more trades before validation (50 minimum)")
    else:
        print("  ✅ Minimum trade count reached for validation")

    # Worst symbol
    worst_sym = min(by_symbol.keys(), key=lambda s: sum(t["pnl"] for t in by_symbol[s]))
    worst_pnl = sum(t["pnl"] for t in by_symbol[worst_sym])
    if worst_pnl < 0:
        print(f"  ⚠️  Worst symbol: {worst_sym} (${worst_pnl:,.2f}) - consider removing")

    # Best symbol
    best_sym = max(by_symbol.keys(), key=lambda s: sum(t["pnl"] for t in by_symbol[s]))
    best_pnl = sum(t["pnl"] for t in by_symbol[best_sym])
    if best_pnl > 0:
        print(f"  ✅ Best symbol: {best_sym} (${best_pnl:,.2f})")

    print("\n" + "=" * 70)


def analyze_logs():
    """Analyze log file for patterns"""
    if not LOG_FILE.exists():
        print("No log file found.")
        return

    print("\n" + "=" * 70)
    print("  LOG FILE ANALYSIS")
    print("=" * 70)

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  Total log lines: {len(lines)}")
    print(f"  Log file size: {LOG_FILE.stat().st_size / 1024:.1f} KB")

    # Count by level
    levels = defaultdict(int)
    for line in lines:
        for level in ["ERROR", "WARNING", "INFO", "DEBUG"]:
            if f"| {level} |" in line:
                levels[level] += 1
                break

    print(f"\n  Log Levels:")
    for level in ["ERROR", "WARNING", "INFO", "DEBUG"]:
        if levels[level]:
            print(f"    {level}: {levels[level]}")

    # Count errors by type
    errors = defaultdict(int)
    for line in lines:
        if "ERROR" in line:
            if "insufficient buying power" in line:
                errors["Insufficient buying power"] += 1
            elif "yfinance" in line.lower() or "delisted" in line:
                errors["yfinance data error"] += 1
            elif "Order failed" in line:
                errors["Order failed"] += 1
            elif "Failed to initialize" in line:
                errors["Initialization error"] += 1
            else:
                errors["Other"] += 1

    if errors:
        print(f"\n  Error Breakdown:")
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {err}: {count}")

    # Count breakout detections
    breakouts = defaultdict(int)
    for line in lines:
        if "BREAKOUT DETECTED" in line:
            for sym in ["SPY", "QQQ", "NVDA", "TSLA", "MSFT", "TSM", "JNJ", "CRDO"]:
                if sym in line:
                    breakouts[sym] += 1

    if breakouts:
        print(f"\n  Breakout Detections (symbols blacklisted):")
        for sym, count in sorted(breakouts.items(), key=lambda x: -x[1]):
            print(f"    {sym}: {count} times")

    # Count signals generated
    signals = defaultdict(int)
    for line in lines:
        if "Generated signal" in line or "Signal detected" in line:
            for sym in ["SPY", "QQQ", "NVDA", "TSLA", "MSFT", "TSM", "JNJ", "CRDO"]:
                if sym in line:
                    signals[sym] += 1

    if signals:
        print(f"\n  Signals Generated:")
        for sym, count in sorted(signals.items(), key=lambda x: -x[1]):
            print(f"    {sym}: {count}")

    # Bot starts/stops
    starts = sum(1 for l in lines if "Box Trading Bot Starting" in l)
    stops = sum(1 for l in lines if "Box Trading Bot stopped" in l)
    print(f"\n  Bot Starts: {starts}  |  Stops: {stops}")

    print("=" * 70)


def main():
    print("\n")
    print("*" * 70)
    print("  BOX TRADING BOT - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("*" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 70)

    # Analyze trade journal
    journal = load_journal()
    if journal:
        analyze_trades(journal)

    # Analyze logs
    analyze_logs()

    print(f"\n  Files analyzed:")
    print(f"    Trade Journal: {TRADE_JOURNAL}")
    print(f"    Log File: {LOG_FILE}")
    print()


if __name__ == "__main__":
    main()
