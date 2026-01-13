from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def filter_date_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ts_col: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # --- Detect datetime column automatically if 'timestamp' is missing ---
    if ts_col not in df.columns:
        for cand in ["date", "datetime", "time", "Date", "Timestamp"]:
            if cand in df.columns:
                ts_col = cand
                break
        else:
            # No recognized datetime column → return df as-is
            return df

    # Convert if needed
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    mask = (df[ts_col] >= start) & (df[ts_col] <= end)
    return df.loc[mask].copy()


def filter_symbols(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns or not symbols:
        return df
    return df[df["symbol"].isin(symbols)].copy()

def daily_pnl_aggregate(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily PnL with automatic column detection."""
    if pnl_df is None or pnl_df.empty:
        return pnl_df

    df = pnl_df.copy()

    # Auto-detect timestamp column
    time_col = None
    for cand in ["timestamp", "date", "datetime", "time", "Date", "Timestamp"]:
        if cand in df.columns:
            time_col = cand
            break

    # Auto-detect pnl column
    pnl_col = None
    for cand in ["day_pnl", "daily_pnl", "pnl", "PnL", "realized_pnl", "profit", "profit_loss"]:
        if cand in df.columns:
            pnl_col = cand
            break

    if not time_col or not pnl_col:
        # Return an empty frame instead of raising
        return pd.DataFrame(columns=["timestamp", "day_pnl", "cum_pnl"])

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.rename(columns={time_col: "timestamp", pnl_col: "day_pnl"})

    out = df.groupby("timestamp", as_index=False)["day_pnl"].sum()
    out["cum_pnl"] = out["day_pnl"].cumsum()
    return out

def equity_stats(equity_df: pd.DataFrame, drawdown_window: int = 252) -> Dict[str, Any]:
    """
    Computes basic equity metrics (current equity, peak, trough, drawdown)
    with automatic column detection so it never KeyErrors.
    """
    if equity_df is None or equity_df.empty:
        return {"equity_now": None, "peak": None, "trough": None, "max_drawdown": None}

    df = equity_df.copy()

    # ---- Auto-detect time column ----
    time_col = None
    for cand in ["timestamp", "date", "datetime", "time", "Date", "Timestamp"]:
        if cand in df.columns:
            time_col = cand
            break

    # ---- Auto-detect equity/value column ----
    val_col = None
    for cand in ["equity", "Equity", "value", "Value", "balance", "Balance", "portfolio_value"]:
        if cand in df.columns:
            val_col = cand
            break

    if not time_col or not val_col:
        # Fallback: return placeholders so Streamlit can render gracefully
        return {"equity_now": None, "peak": None, "trough": None, "max_drawdown": None}

    # ---- Core logic ----
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, val_col]).sort_values(time_col)

    if df.empty:
        return {"equity_now": None, "peak": None, "trough": None, "max_drawdown": None}

    eq_series = df[val_col].astype(float)
    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max

    return {
        "equity_now": float(eq_series.iloc[-1]),
        "peak": float(roll_max.max()),
        "trough": float(eq_series.loc[drawdown.idxmin()]),
        "max_drawdown": float(drawdown.min()),
    }


def rolling_sharpe_from_pnl(pnl_daily: pd.DataFrame, window_days: int = 30, eps: float = 1e-9) -> pd.DataFrame:
    """Compute rolling Sharpe ratio (tolerant to missing or renamed columns)."""
    if pnl_daily is None or pnl_daily.empty:
        return pnl_daily

    df = pnl_daily.copy()

    # Detect proper column
    ret_col = None
    for cand in ["day_pnl", "ret", "pnl", "daily_pnl", "realized_pnl"]:
        if cand in df.columns:
            ret_col = cand
            break
    if not ret_col:
        return df  # skip safely

    df["ret"] = df[ret_col].astype(float)
    roll_mean = df["ret"].rolling(window_days).mean()
    roll_std = df["ret"].rolling(window_days).std(ddof=0)
    df["rolling_sharpe"] = (roll_mean / (roll_std + eps)) * np.sqrt(252)
    return df

def latest_symbol_weights(drift_df: pd.DataFrame) -> pd.DataFrame:
    if drift_df is None or drift_df.empty:
        return drift_df
    last_ts = drift_df["timestamp"].max()
    snap = drift_df[drift_df["timestamp"] == last_ts].copy()
    # Normalize to 1.0 just in case
    total = snap["weight"].sum()
    if total and total != 0:
        snap["weight"] = snap["weight"] / total
    return snap.sort_values("weight", ascending=False)
