from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from .metrics_io import normalize_columns

def _pick_cols(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _ensure_min_rows(df: pd.DataFrame, min_rows: int = 10) -> bool:
    return df is not None and not df.empty and len(df) >= min_rows

def _annualize_vol(ret: pd.Series) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty: return 0.0
    return float(ret.std(ddof=0) * np.sqrt(252))

def _sharpe(ret: pd.Series, eps: float = 1e-9) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty: return 0.0
    return float((ret.mean() / (ret.std(ddof=0) + eps)) * np.sqrt(252))

def _sortino(ret: pd.Series, eps: float = 1e-9) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty: return 0.0
    downside = ret[ret < 0]
    ds = downside.std(ddof=0)
    return float((ret.mean() / (ds + eps)) * np.sqrt(252))

def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    roll = equity.cummax()
    dd = (equity - roll) / roll
    return float(dd.min())

def _trend_r2(ts: pd.Series, y: pd.Series) -> float:
    """
    R^2 of equity vs time trend. Gives 0..1 measure of smoothness/consistency.
    """
    if ts is None or y is None or len(ts) < 5:
        return 0.0
    # Convert timestamp to ordinal float
    x = pd.to_datetime(ts, errors="coerce").astype("int64") / 1e9  # seconds
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.pinv(X) @ y.values
    y_hat = X @ beta
    ss_res = np.sum((y.values - y_hat) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2) + 1e-9
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(0.0, min(1.0, r2)))

def _scale_0_100(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """
    Linear scale to 0..100 within [lo, hi]. Clamp outside.
    If invert=True, higher x -> lower score.
    """
    if hi == lo:
        return 50.0
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    if invert:
        t = 1.0 - t
    return t * 100.0

def compute_risk_grade(equity_df: pd.DataFrame, pnl_df: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
    """
    Robust evaluator:
      - Normalizes columns.
      - Filters to last N days.
      - Computes Sharpe, Sortino, MaxDD, ann. vol, 30d PnL and equity trend R^2.
      - Produces composite score and letter grade.
    """
    equity_df = normalize_columns(equity_df)
    pnl_df = normalize_columns(pnl_df)

    # Basic column picks
    tcol = _pick_cols(equity_df, ["timestamp"])
    ecol = _pick_cols(equity_df, ["equity"])
    p_tcol = _pick_cols(pnl_df, ["timestamp"])
    p_col = _pick_cols(pnl_df, ["day_pnl"])

    # Early exit if missing
    if not tcol or not ecol or not p_tcol or not p_col:
        return {"grade": "N/A", "score": 0.0, "msg": "insufficient columns"}

    # Filter date window
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)

    eq = equity_df.copy()
    eq = eq[(eq[tcol] >= start) & (eq[tcol] <= end)]
    pnl = pnl_df.copy()
    pnl = pnl[(pnl[p_tcol] >= start) & (pnl[p_tcol] <= end)]

    if not _ensure_min_rows(eq, 10) or not _ensure_min_rows(pnl, 5):
        return {"grade": "N/A", "score": 0.0, "msg": "insufficient rows"}

    # Returns from equity
    eq_sorted = eq.sort_values(tcol)
    ret = eq_sorted[ecol].pct_change()  # daily returns

    vol_ann = _annualize_vol(ret)
    sharpe = _sharpe(ret)
    sortino = _sortino(ret)
    max_dd = _max_drawdown(eq_sorted[ecol])

    pnl_30d = float(pd.to_numeric(pnl[p_col], errors="coerce").dropna().sum())
    r2 = _trend_r2(eq_sorted[tcol], pd.to_numeric(eq_sorted[ecol], errors="coerce").fillna(method="ffill").dropna())

    # ------ Composite scoring (0..100) ------
    # Calibrations (tweak as you like)
    sharpe_score = _scale_0_100(sharpe, lo=0.0, hi=2.0)           # Sharpe 0..2+
    dd_score     = _scale_0_100(max_dd, lo=-0.30, hi=0.0, invert=True)  # -30%..0%
    pnl_score    = _scale_0_100(pnl_30d, lo=0.0, hi=5000.0)       # $0..$5k
    vol_score    = _scale_0_100(vol_ann, lo=0.05, hi=0.40, invert=True) # 5%..40%
    r2_score     = _scale_0_100(r2, lo=0.2, hi=0.95)              # 0.2..0.95

    score = 0.40 * sharpe_score + 0.25 * dd_score + 0.15 * pnl_score + 0.10 * vol_score + 0.10 * r2_score
    score = float(round(score, 1))

    grade = (
        "A+" if score >= 92 else
        "A"  if score >= 85 else
        "A-" if score >= 80 else
        "B+" if score >= 75 else
        "B"  if score >= 70 else
        "C"  if score >= 60 else
        "D"  if score >= 50 else
        "E"
    )

    return {
        "window_days": days,
        "equity_now": float(eq_sorted[ecol].iloc[-1]),
        "metrics": {
            "sharpe": round(float(sharpe), 3),
            "sortino": round(float(sortino), 3),
            "vol_ann": round(float(vol_ann), 3),
            "max_drawdown": round(float(max_dd), 4),
            "pnl_sum": round(float(pnl_30d), 2),
            "trend_r2": round(float(r2), 3),
        },
        "score": score,
        "grade": grade,
        "decision": _decision_from_grade(grade, max_dd),
    }

def _decision_from_grade(grade: str, max_dd: float) -> str:
    """
    Human-readable policy hook used by later phases (allocator/guardian).
    """
    if grade in ("A+", "A"):
        return "✅ Allow scaling +5%"
    if grade in ("A-", "B+"):
        return "✅ Maintain size; selective scaling"
    if grade == "B":
        return "🟨 Maintain; tighten risk if volatility rises"
    if grade == "C":
        return "🟧 Reduce exposure -10%; monitor drawdown"
    if grade == "D":
        return "🟥 Reduce exposure -20%; freeze new positions"
    # E or anything else
    return "⛔ Halt scaling; de-risk portfolio"
