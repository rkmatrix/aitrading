"""
Phase 21.1 — Live Ledger → Weekly PnL Bridge
Appends closed trades from RealTimeExecutionAdapter into ledger & weekly summary.
"""

from __future__ import annotations
import csv, os, datetime as dt, pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

LEDGER_PATH = Path("data/trades/ledger.csv")
WEEKLY_PATH = Path("data/pnl/weekly_pnl.csv")


def _ensure_dirs():
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEEKLY_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_fill(symbol: str, pnl: float, entry_price: float, exit_price: float,
             opened_at: dt.datetime, closed_at: dt.datetime, duration_s: int):
    """Append closed trade info to CSV ledger and roll up weekly PnL."""
    _ensure_dirs()
    week_ending = (closed_at + dt.timedelta(days=(6 - closed_at.weekday()))).strftime("%Y-%m-%d")
    row = {
        "timestamp": closed_at.isoformat(sep=" ", timespec="seconds"),
        "symbol": symbol,
        "pnl": round(pnl, 2),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "opened_at": opened_at.isoformat(sep=" ", timespec="seconds"),
        "closed_at": closed_at.isoformat(sep=" ", timespec="seconds"),
        "duration_s": duration_s,
        "week_ending": week_ending,
    }

    # --- append to ledger.csv ---
    write_header = not LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # --- update weekly_pnl.csv ---
    try:
        df = pd.read_csv(WEEKLY_PATH) if WEEKLY_PATH.exists() else pd.DataFrame(columns=["week_ending", "total_pnl"])
        week_mask = df["week_ending"] == week_ending if not df.empty else []
        if week_mask.any():
            df.loc[week_mask, "total_pnl"] += row["pnl"]
        else:
            df.loc[len(df)] = [week_ending, row["pnl"]]
        df.to_csv(WEEKLY_PATH, index=False)
    except Exception as e:
        log.warning("Could not update weekly PnL: %s", e)
