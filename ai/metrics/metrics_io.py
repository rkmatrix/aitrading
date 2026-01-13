from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np

# ---------- Column normalization helpers ----------

# Canonical column names we want to end up with
# timestamp, equity, day_pnl, symbol, weight
ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "date", "datetime", "time", "Date", "Timestamp"],
    "equity": ["equity", "Equity", "value", "Value", "balance", "Balance",
               "internal_equity", "broker_equity", "portfolio_value"],
    "day_pnl": ["day_pnl", "daily_pnl", "pnl", "PnL", "realized_pnl", "profit", "profit_loss"],
    "symbol": ["symbol", "ticker", "asset", "sym"],
    "weight": ["weight", "w", "alloc", "allocation", "pct", "drift_pct"],
}

def _find_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Return a copy with best-effort normalization to:
      timestamp, equity, day_pnl, symbol, weight
    Any missing columns are simply not created (no crash).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    mapping = {}
    for canon, cands in ALIASES.items():
        src = _find_first_present(out, cands)
        if src and src != canon:
            mapping[src] = canon
        elif src is None:
            # leave missing; we don't fabricate values here
            pass
    if mapping:
        out = out.rename(columns=mapping)

    # Coerce timestamp if present
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Coerce numerics if present
    for col in ["equity", "day_pnl", "weight"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "weight" in out.columns:
        # If weights look like percents (0..100), normalize to 0..1 by heuristic
        if out["weight"].dropna().max() > 1.5:
            out["weight"] = out["weight"] / 100.0

    return out

# ---------- CSV loading ----------

def _coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # If column not found, try common alternatives
    if col not in df.columns:
        for alt in ["date", "datetime", "time", "Date", "Timestamp"]:
            if alt in df.columns:
                col = alt
                break
        else:
            return df  # no date-like column at all
    df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")
    df = df.dropna(subset=[col])
    df = df.sort_values(col)
    return df

def load_csv_if_exists(path: Path, parse_dates: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if parse_dates:
        df = _coerce_datetime(df, parse_dates)
    return df

def try_load_reports(pnl_path: Path, equity_path: Path, drift_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    pnl_df = load_csv_if_exists(pnl_path, parse_dates="timestamp")
    equity_df = load_csv_if_exists(equity_path, parse_dates="timestamp")
    drift_df = load_csv_if_exists(drift_path, parse_dates="timestamp")

    # Fallback: some older phases may name drift file differently
    if drift_df is None:
        alt = drift_path.parent / "equity_drift.csv"
        drift_df = load_csv_if_exists(alt, parse_dates="timestamp")

    # Normalize column names for downstream safety
    pnl_df = normalize_columns(pnl_df)
    equity_df = normalize_columns(equity_df)
    drift_df = normalize_columns(drift_df)
    return pnl_df, equity_df, drift_df

# ---------- Demo generation to keep UIs & tests alive ----------

def ensure_demo_if_missing(pnl_df, equity_df, drift_df):
    """
    If any dataset is missing, generate small deterministic demo frames so the
    pipeline remains usable for layout/testing.
    """
    rng = np.random.default_rng(1234)
    if pnl_df is None or pnl_df.empty:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90, freq="B")
        pnl_df = pd.DataFrame({
            "timestamp": idx,
            "symbol": rng.choice(["AAPL", "MSFT", "TSLA"], size=len(idx)),
            "day_pnl": rng.normal(0, 150, size=len(idx)).round(2),
        })

    if equity_df is None or equity_df.empty:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120, freq="B")
        trend = np.linspace(0, 120*120, len(idx))  # slow uptrend
        noise = rng.normal(0, 400, size=len(idx)).cumsum()
        equity = 100_000 + trend + noise
        equity_df = pd.DataFrame({"timestamp": idx, "equity": equity.round(2)})

    if drift_df is None or drift_df.empty:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="B")
        symbols = ["AAPL", "MSFT", "TSLA"]
        rows = []
        for t in idx:
            raw = np.abs(rng.normal(1.0, 0.4, size=len(symbols)))
            raw = raw / raw.sum()
            for s, w in zip(symbols, raw):
                rows.append({"timestamp": t, "symbol": s, "weight": round(float(w), 4)})
        drift_df = pd.DataFrame(rows)

    # Ensure normalized names
    pnl_df = normalize_columns(pnl_df)
    equity_df = normalize_columns(equity_df)
    drift_df = normalize_columns(drift_df)

    return pnl_df, equity_df, drift_df
