# ai/features/feature_builder.py
# -*- coding: utf-8 -*-
"""
Phase 8.2 — Feature Builder (single-symbol + cross-symbol features)

This module builds robust, model-friendly per-symbol features and merges
CrossSymbolIntel outputs (correlations, sector strength, relative momentum, etc.).

Inputs:
- price_panel: wide DataFrame, index=Datetime, columns=tickers, values=Close (adj or not)
- optional ohlcv_panel: dict of {symbol: DataFrame with columns [Open,High,Low,Close,Volume]}
- sectors_map: {symbol: sector_str}
- intel: CrossSymbolIntel instance (already fit on price_panel)

Outputs:
- features_df: DataFrame indexed by symbol (latest snapshot), numeric columns ready for ML
- meta: dict with helper metadata

Design notes:
- Strong defensive handling for (n,1) shapes, tz-naive/aware, missing values.
- Only uses lightweight tech features to avoid heavy dependencies.

Author: AITradeBot Phase 8.2
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

NUMERIC_EPS = 1e-12


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _squeeze_series(x) -> pd.Series:
    """Return a 1-D pandas Series from any column-like object safely."""
    s = pd.Series(x).copy()
    # If it's a DataFrame or 2D, take first column
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
    return pd.to_numeric(pd.Series(s).squeeze(), errors="coerce")


def _pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    s = _squeeze_series(s)
    return s.pct_change(periods=periods)


def _zscore(s: pd.Series, lookback: int) -> float:
    s = _squeeze_series(s).tail(lookback).dropna()
    if len(s) < 3:
        return np.nan
    mu, sd = s.mean(), s.std()
    if sd <= NUMERIC_EPS or np.isnan(sd):
        return np.nan
    return float((s.iloc[-1] - mu) / sd)


def _sma(s: pd.Series, window: int) -> float:
    s = _squeeze_series(s).rolling(window, min_periods=max(2, window // 2)).mean()
    return float(s.iloc[-1]) if len(s) else np.nan


def _ema(s: pd.Series, span: int) -> float:
    s = _squeeze_series(s).ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()
    return float(s.iloc[-1]) if len(s) else np.nan


def _rsi(close: pd.Series, period: int = 14) -> float:
    c = _squeeze_series(close)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down + NUMERIC_EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = rsi.iloc[-1] if len(rsi) else np.nan
    return float(val)


def _atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    if ohlcv is None or ohlcv.empty:
        return np.nan
    df = ohlcv.copy()
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            return np.nan
    df = _ensure_dtindex(df)
    high = _squeeze_series(df["High"])
    low = _squeeze_series(df["Low"])
    close = _squeeze_series(df["Close"])
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return float(atr.iloc[-1]) if len(atr) else np.nan


def build_features(
    price_panel: pd.DataFrame,
    ohlcv_panel: Optional[Dict[str, pd.DataFrame]],
    sectors_map: Dict[str, str],
    intel,  # CrossSymbolIntel instance already fit
    *,
    lookbacks: Tuple[int, int, int] = (5, 20, 60),
) -> Tuple[pd.DataFrame, dict]:
    """
    Build the latest-snapshot features per symbol and merge CrossSymbolIntel features.

    Returns:
        features_df: index=symbol, numeric columns ready for model
        meta: dict with lookbacks and available columns
    """
    if price_panel is None or price_panel.empty:
        raise ValueError("price_panel is empty")

    price_panel = _ensure_dtindex(price_panel).ffill().bfill()
    symbols = list(price_panel.columns)

    lb_short, lb_mid, lb_long = lookbacks

    rows = []
    for sym in symbols:
        close = _squeeze_series(price_panel[sym])

        feat = {
            "symbol": sym,
            # Simple momentum ratios
            f"ret_{lb_short}": float(_pct_change(close, lb_short).iloc[-1]) if len(close) > lb_short else np.nan,
            f"ret_{lb_mid}": float(_pct_change(close, lb_mid).iloc[-1]) if len(close) > lb_mid else np.nan,
            f"ret_{lb_long}": float(_pct_change(close, lb_long).iloc[-1]) if len(close) > lb_long else np.nan,
            # Trend proxies
            f"sma_{lb_short}": _sma(close, lb_short),
            f"sma_{lb_mid}": _sma(close, lb_mid),
            f"ema_{lb_short}": _ema(close, lb_short),
            f"ema_{lb_mid}": _ema(close, lb_mid),
            # Z-score (long lookback)
            f"zscore_{lb_mid}": _zscore(close, lb_mid),
            f"zscore_{lb_long}": _zscore(close, lb_long),
            # RSI
            "rsi_14": _rsi(close, 14),
            # Sector
            "sector": sectors_map.get(sym, "UNKNOWN"),
        }

        # ATR from OHLCV if available
        atr_val = np.nan
        if ohlcv_panel and sym in ohlcv_panel:
            atr_val = _atr(ohlcv_panel[sym], period=14)
        feat["atr_14"] = atr_val

        rows.append(feat)

    feats_df = pd.DataFrame(rows).set_index("symbol")
    # Merge CrossSymbolIntel outputs (pre-computed)
    if intel is not None and intel.features_df is not None and not intel.features_df.empty:
        # Avoid column collisions; prefer intel naming as-is
        feats_df = feats_df.join(intel.features_df, how="left")

    # Numeric cleaning
    feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

    meta = {
        "lookbacks": lookbacks,
        "columns": list(feats_df.columns),
        "intel_columns": list(intel.features_df.columns) if intel and intel.features_df is not None else [],
    }
    return feats_df, meta
