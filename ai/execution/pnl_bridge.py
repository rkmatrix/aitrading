# -*- coding: utf-8 -*-
"""
Phase 20 → Phase 17.5 PnL Tracker Bridge
Collects execution-level metrics and writes to the central PnL store.
"""
import os
import datetime as dt
import pandas as pd
from pathlib import Path
from tools.telegram_alerts import send_weekly_summary

# Directory where weekly_pnl.csv or DB is stored
PNL_PATH = Path("data/pnl/weekly_pnl.csv")

def _load_pnl():
    if PNL_PATH.exists():
        return pd.read_csv(PNL_PATH)
    return pd.DataFrame(columns=["date", "symbol", "side", "pnl", "slippage_bps", "trades"])

def _save_pnl(df: pd.DataFrame):
    PNL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PNL_PATH, index=False)

def record_execution(symbol: str, side: str, pnl: float, slippage_bps: float, trades: int = 1):
    df = _load_pnl()
    today = dt.date.today().isoformat()
    row = pd.DataFrame([{
        "date": today,
        "symbol": symbol,
        "side": side,
        "pnl": pnl,
        "slippage_bps": slippage_bps,
        "trades": trades,
    }])
    df = pd.concat([df, row], ignore_index=True)
    _save_pnl(df)
    print(f"[PnLTracker] Recorded {symbol} {side} → {pnl:.2f} ({slippage_bps:.2f} bps)")

def summarize_weekly():
    df = _load_pnl()
    if df.empty:
        print("No PnL data yet.")
        return None
    df["date"] = pd.to_datetime(df["date"])
    this_week = df[df["date"] >= (pd.Timestamp.today() - pd.Timedelta(days=7))]
    total_pnl = this_week["pnl"].sum()
    total_trades = this_week["trades"].sum()
    send_weekly_summary(total_pnl, total_trades)
    print(f"[PnLTracker] Weekly Summary → Trades={total_trades}, PnL={total_pnl:.2f}")
    return total_pnl, total_trades
