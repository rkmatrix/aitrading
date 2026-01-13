# ai/analytics/cross_symbol_intel.py
# -*- coding: utf-8 -*-
"""
Phase 8.2 — Cross-Symbol Intelligence Layer: Correlation Matrix Engine

This module computes rolling inter-symbol correlations, sector-level strength,
relative momentum, and pair z-scores. It exposes a single orchestrator class
`CrossSymbolIntel` with pure, dependency-light methods designed to plug into
your Phase 8 portfolio brain and feature pipeline.

Key outputs:
- correlation_matrix: pd.DataFrame (symbols x symbols) for latest window
- features_df:       pd.DataFrame (index=symbol) with columns:
    ['symbol_corr_mean', 'symbol_corr_rank', 'sector', 'sector_strength',
     'relative_momentum', 'volatility', 'beta_spy(optional)', ...]
- pair_stats:        pd.DataFrame listing top correlated/divergent pairs with z-scores
- history snapshots (optional) to support your learning feedback loop

Conventions/assumptions:
- `prices` is a wide DataFrame of *Close* (or adjusted close) prices with
   index=Datetime (tz-aware or tz-naive ok; we localize consistently) and
   columns = tickers (strings).
- Missing/irregular data is handled via forward-fill then back-fill, then
  optional rolling min periods for statistics.
- Sectors mapping is a dict: { 'AAPL': 'TECH', 'MSFT': 'TECH', 'XOM': 'ENERGY', ... }
  Sector strength is computed as the average momentum of symbols in that sector.

Robustness:
- Squeezes 2D Series/arrays defensively to avoid "Data must be 1-dimensional" errors.
- Handles yfinance auto_adjust changes (you pass in prices; we don't fetch here).
- Works on Windows/Python 3.11 and common pandas versions.

Usage (programmatic):
    intel = CrossSymbolIntel(window=20, mom_lookback=20, vol_lookback=20)
    intel.fit(prices, sectors_map)
    corr = intel.correlation_matrix
    feats = intel.features_df
    pairs = intel.pair_stats

CLI quick test:
    python -m ai.analytics.cross_symbol_intel

Author: AITradeBot Phase 8.2
"""

from __future__ import annotations

import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------- Logging ---------------------------------------------

logger = logging.getLogger("cross_symbol_intel")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ----------------------- Utilities -------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError("prices index must be datetime-like") from e
    return df


def _sanitize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure wide prices DF:
      - Columns: symbols (str)
      - Index: DatetimeIndex
      - Numeric dtype
      - Sorted index
      - FFill/BFill to patch small gaps, then drop rows fully NaN.
    """
    if prices is None or prices.empty:
        raise ValueError("`prices` is empty")

    df = _ensure_datetime_index(prices)
    df = df.sort_index()
    # Coerce to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    # Forward then back fill small gaps
    df = df.ffill().bfill()
    # Drop rows that are still all NaN
    df = df.dropna(how="all")
    # Drop columns that are all NaN
    df = df.dropna(axis=1, how="all")

    # Squeeze any 2D to 1D within columns (guard against (n,1) arrays)
    for c in df.columns:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            df[c] = col.iloc[:, 0]
    return df


def _safe_rank(series: pd.Series) -> pd.Series:
    # Percentile rank; if all equal/NaN handles gracefully
    s = series.astype(float)
    return s.rank(pct=True, method="average")


def _rolling_corr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute correlation matrix for the *last* window of returns.
    We compute log returns to stabilize variance.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    # log returns
    rets = np.log(df / df.shift(1))
    recent = rets.tail(window)
    if recent.dropna(how="all").empty:
        # fallback: try simple pct_change if log returns are degenerate
        recent = df.pct_change().tail(window)

    corr = recent.corr(method="pearson")
    return corr


def _momentum(prices: pd.Series, lookback: int) -> float:
    """
    Simple momentum: price_t / price_(t-LB) - 1
    Returns NaN if insufficient history.
    """
    s = pd.Series(prices).astype(float)
    s = s.squeeze()
    if len(s) <= lookback or lookback <= 0:
        return np.nan
    try:
        last = float(s.iloc[-1])
        prev = float(s.iloc[-lookback - 1])
        if prev == 0 or np.isnan(last) or np.isnan(prev):
            return np.nan
        return (last / prev) - 1.0
    except Exception:
        return np.nan


def _volatility(prices: pd.Series, lookback: int) -> float:
    """
    Realized volatility: std of log returns over lookback, annualized ~sqrt(252)
    """
    s = pd.Series(prices).astype(float).squeeze()
    if len(s) <= lookback or lookback <= 1:
        return np.nan
    lr = np.log(s / s.shift(1))
    recent = lr.tail(lookback).dropna()
    if len(recent) < 2:
        return np.nan
    return float(recent.std() * np.sqrt(252))


def _pair_zscore(a: pd.Series, b: pd.Series, lookback: int) -> float:
    """
    Z-score of the price spread between a and b over lookback.
    Uses log prices for stability. Returns NaN if insufficient data.
    """
    a = pd.Series(a).astype(float).squeeze()
    b = pd.Series(b).astype(float).squeeze()
    if len(a) < lookback + 5 or len(b) < lookback + 5:
        return np.nan

    la = np.log(a).tail(lookback)
    lb = np.log(b).tail(lookback)
    spread = la - lb
    mu, sd = spread.mean(), spread.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((spread.iloc[-1] - mu) / sd)


# ----------------------- Main Orchestrator -----------------------------------

@dataclass
class CrossSymbolIntel:
    window: int = 20              # rolling corr window
    mom_lookback: int = 20        # momentum lookback
    vol_lookback: int = 20        # realized vol lookback
    pair_min_corr: float = 0.6    # filter for "highly correlated" pairs
    top_k_pairs: int = 10         # how many pairs to keep for reporting
    base_currency: Optional[str] = None  # optional: e.g., 'SPY' for beta
    snapshots_dir: Optional[str] = None  # optional: write JSON/CSV snapshots

    # Computed artifacts
    correlation_matrix: Optional[pd.DataFrame] = field(default=None, init=False)
    features_df: Optional[pd.DataFrame] = field(default=None, init=False)
    pair_stats: Optional[pd.DataFrame] = field(default=None, init=False)

    def fit(self, prices: pd.DataFrame, sectors_map: Optional[Dict[str, str]] = None) -> "CrossSymbolIntel":
        """
        Compute correlations and cross-symbol features. Stores results on self.
        """
        logger.info("Starting CrossSymbolIntel.fit(...)")
        prices = _sanitize_prices(prices)

        # Align sector map
        if sectors_map is None:
            sectors_map = {}
        sectors_map = {sym: sectors_map.get(sym, "UNKNOWN") for sym in prices.columns}

        # 1) Correlation matrix
        corr = _rolling_corr(prices, self.window)
        self.correlation_matrix = corr

        # 2) Per-symbol stats
        feats = self._build_symbol_features(prices, corr, sectors_map)

        # 3) Pair stats (for reporting / pairs trading ideas)
        pairs = self._build_pair_stats(prices, corr)

        self.features_df = feats
        self.pair_stats = pairs

        # 4) Optional snapshot to disk for learning feedback loops
        if self.snapshots_dir:
            self._snapshot_to_disk()

        logger.info("CrossSymbolIntel.fit(...) completed.")
        return self

    # ------------------- Feature Builders ------------------------------------

    def _build_symbol_features(
        self,
        prices: pd.DataFrame,
        corr: pd.DataFrame,
        sectors_map: Dict[str, str],
    ) -> pd.DataFrame:
        symbols = list(prices.columns)

        # symbol mean correlation (exclude self)
        corr_no_diag = corr.copy()
        np.fill_diagonal(corr_no_diag.values, np.nan)
        mean_corr = corr_no_diag.mean(axis=1).fillna(0.0)

        # momentum & volatility
        momentum_vals = {}
        vol_vals = {}
        for sym in symbols:
            s = prices[sym]
            momentum_vals[sym] = _momentum(s, self.mom_lookback)
            vol_vals[sym] = _volatility(s, self.vol_lookback)

        # sector strength: average momentum of symbols in same sector
        sec_strength = {}
        # Precompute per-sector symbol lists
        sec_to_syms: Dict[str, List[str]] = {}
        for sym, sec in sectors_map.items():
            sec_to_syms.setdefault(sec, []).append(sym)

        sym_rel_mom = {}
        for sym in symbols:
            sec = sectors_map.get(sym, "UNKNOWN")
            bucket = sec_to_syms.get(sec, [])
            if len(bucket) == 0:
                sec_strength_val = np.nan
            else:
                vals = [momentum_vals[b] for b in bucket if not np.isnan(momentum_vals[b])]
                sec_strength_val = float(np.mean(vals)) if len(vals) else np.nan
            sec_strength[sym] = sec_strength_val

            # relative momentum: symbol's momentum minus sector avg
            rm = np.nan
            if not np.isnan(momentum_vals[sym]) and not np.isnan(sec_strength_val):
                rm = float(momentum_vals[sym] - sec_strength_val)
            sym_rel_mom[sym] = rm

        # (Optional) beta vs base_currency (e.g., SPY) over the window
        beta_vals = {}
        if self.base_currency and self.base_currency in symbols:
            base = np.log(prices[self.base_currency] / prices[self.base_currency].shift(1)).tail(self.window)
            base = base.dropna()
            for sym in symbols:
                if sym == self.base_currency:
                    beta_vals[sym] = 1.0
                    continue
                r = np.log(prices[sym] / prices[sym].shift(1)).tail(self.window).dropna()
                idx = base.index.intersection(r.index)
                if len(idx) < 5:
                    beta_vals[sym] = np.nan
                    continue
                x = base.loc[idx].values
                y = r.loc[idx].values
                # simple OLS beta = cov(x,y)/var(x)
                vx = np.var(x, ddof=1)
                beta_vals[sym] = float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else np.nan
        else:
            beta_vals = {sym: np.nan for sym in symbols}

        # Assemble feature frame
        df = pd.DataFrame(
            {
                "symbol": symbols,
                "symbol_corr_mean": [mean_corr.get(sym, np.nan) for sym in symbols],
                "symbol_corr_rank": [float(_safe_rank(mean_corr).get(sym, np.nan)) for sym in symbols],
                "sector": [sectors_map.get(sym, "UNKNOWN") for sym in symbols],
                "momentum": [momentum_vals[sym] for sym in symbols],
                "sector_strength": [sec_strength[sym] for sym in symbols],
                "relative_momentum": [sym_rel_mom[sym] for sym in symbols],
                "volatility": [vol_vals[sym] for sym in symbols],
                "beta_vs_base": [beta_vals[sym] for sym in symbols],
            }
        ).set_index("symbol")

        # Cleanups
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _build_pair_stats(self, prices: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Build pair list:
          - Highly correlated pairs (|corr| >= pair_min_corr)
          - Compute pair z-score (spread signal)
          - Rank top-k by |corr|
        """
        syms = list(prices.columns)
        records: List[Tuple[str, str, float, float]] = []

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                c = float(corr.loc[a, b]) if (a in corr.index and b in corr.columns) else np.nan
                if np.isnan(c) or abs(c) < self.pair_min_corr:
                    continue
                z = _pair_zscore(prices[a], prices[b], lookback=max(self.window, self.mom_lookback))
                records.append((a, b, c, z))

        if not records:
            return pd.DataFrame(columns=["sym_a", "sym_b", "corr", "pair_zscore"])

        df = pd.DataFrame(records, columns=["sym_a", "sym_b", "corr", "pair_zscore"])
        # Rank by absolute correlation, keep top-k for readability
        df["abs_corr"] = df["corr"].abs()
        df = df.sort_values(["abs_corr"], ascending=[False]).head(self.top_k_pairs).drop(columns=["abs_corr"])
        return df.reset_index(drop=True)

    # ------------------- Snapshots -------------------------------------------

    def _snapshot_to_disk(self) -> None:
        try:
            os.makedirs(self.snapshots_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create snapshots_dir='{self.snapshots_dir}': {e}")
            return

        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

        # Correlation matrix
        try:
            corr_path = os.path.join(self.snapshots_dir, f"corr_{ts}.csv")
            self.correlation_matrix.to_csv(corr_path)
        except Exception as e:
            logger.warning(f"Failed to write correlation snapshot: {e}")

        # Features
        try:
            feats_path = os.path.join(self.snapshots_dir, f"features_{ts}.csv")
            self.features_df.to_csv(feats_path)
        except Exception as e:
            logger.warning(f"Failed to write features snapshot: {e}")

        # Pair stats
        try:
            pairs_path = os.path.join(self.snapshots_dir, f"pairs_{ts}.csv")
            self.pair_stats.to_csv(pairs_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to write pairs snapshot: {e}")

        # Small JSON meta
        try:
            meta = {
                "window": self.window,
                "mom_lookback": self.mom_lookback,
                "vol_lookback": self.vol_lookback,
                "pair_min_corr": self.pair_min_corr,
                "top_k_pairs": self.top_k_pairs,
                "base_currency": self.base_currency,
                "features_columns": list(self.features_df.columns) if self.features_df is not None else [],
            }
            meta_path = os.path.join(self.snapshots_dir, f"meta_{ts}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write meta snapshot: {e}")

    # ------------------- Public Helpers --------------------------------------

    def get_symbol_feature_row(self, symbol: str) -> Optional[pd.Series]:
        if self.features_df is None or symbol not in self.features_df.index:
            return None
        return self.features_df.loc[symbol].copy()

    def to_dict(self) -> Dict[str, dict]:
        """
        Export per-symbol features as plain dict for downstream (e.g., feature_builder).
        """
        if self.features_df is None or self.features_df.empty:
            return {}
        out = {}
        for sym, row in self.features_df.iterrows():
            out[sym] = {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in row.items()}
        return out


# ----------------------- CLI Test Harness ------------------------------------

def _demo_prices() -> pd.DataFrame:
    """
    Build a tiny synthetic price panel for quick smoke testing without I/O.
    """
    rng = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=120, freq="B")
    np.random.seed(42)

    def sim_walk(n=120, drift=0.0005, vol=0.01, start=100.0):
        steps = np.random.normal(drift, vol, n)
        return start * np.exp(np.cumsum(steps))

    data = {
        "AAPL": sim_walk(len(rng), drift=0.0006),
        "MSFT": sim_walk(len(rng), drift=0.00055),
        "GOOG": sim_walk(len(rng), drift=0.0005),
        "XOM":  sim_walk(len(rng), drift=0.0003),
        "CVX":  sim_walk(len(rng), drift=0.00028),
        "SPY":  sim_walk(len(rng), drift=0.0004),
    }
    df = pd.DataFrame(data, index=rng)
    return df


def _demo_sectors(symbols: Iterable[str]) -> Dict[str, str]:
    mapping = {}
    for s in symbols:
        if s in {"AAPL", "MSFT", "GOOG"}:
            mapping[s] = "TECH"
        elif s in {"XOM", "CVX"}:
            mapping[s] = "ENERGY"
        elif s in {"SPY"}:
            mapping[s] = "INDEX"
        else:
            mapping[s] = "UNKNOWN"
    return mapping


def main():
    # Smoke test: run without any external data
    prices = _demo_prices()
    sectors = _demo_sectors(prices.columns)

    intel = CrossSymbolIntel(
        window=20,
        mom_lookback=20,
        vol_lookback=20,
        pair_min_corr=0.4,
        top_k_pairs=8,
        base_currency="SPY",
        snapshots_dir=None,  # set to "artifacts/cross_symbol" to write snapshots
    ).fit(prices, sectors)

    print("\n=== Correlation Matrix (tail) ===")
    print(intel.correlation_matrix.tail(3))

    print("\n=== Features (head) ===")
    print(intel.features_df.head())

    print("\n=== Pair Stats ===")
    print(intel.pair_stats)


if __name__ == "__main__":
    main()
