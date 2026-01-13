# ai/allocators/cross_symbol_intel.py
# ===============================================================
#  Cross-Symbol Intelligence Layer (Phase 8.2)
#  - Clusters symbols by rolling correlation (graph thresholding)
#  - Enforces per-cluster allocation caps (e.g., 30%)
#  - Tightens caps automatically in high-vol / high-correlation regimes
#
#  Zero external deps beyond numpy/pandas. yfinance only as optional fallback
#  if your project doesn't expose utils.market_data.get_price_history.
# ===============================================================

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


class CrossSymbolAllocator:
    """
    Build correlation-based clusters and apply cluster caps to target weights.

    Pipeline:
      1) Fetch rolling prices -> returns -> correlation matrix
      2) Build graph where edge if |corr| >= corr_threshold
      3) Connected components => clusters
      4) Detect regime via average abs-corr & cross-sectional volatility
      5) Apply per-cluster cap (reduced further in high-vol/corr regimes)
      6) Proportionally downscale overweight clusters, optionally redistribute excess

    Notes:
      - No sklearn/scipy required. Graph via adjacency + DFS.
      - Fetch prices via your own market_data util if available.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback_days: int = 60,
        corr_threshold: float = 0.70,
        base_cluster_cap: float = 0.30,       # 30% per cluster baseline
        high_corr_threshold: float = 0.55,    # avg |corr| above => "high corr" regime
        high_vol_threshold: float = 0.018,    # mean daily std above => "high vol" regime (~1.8%)
        cap_tighten_factor: float = 0.6,      # multiply cap in high regimes (e.g., 0.30 -> 0.18)
        recluster_minutes: int = 30,
        state_path: str = "data/cross_symbol_allocator.json",
    ):
        self.symbols = list(dict.fromkeys([s.upper() for s in symbols]))
        self.lookback_days = lookback_days
        self.corr_threshold = float(corr_threshold)
        self.base_cluster_cap = float(base_cluster_cap)
        self.high_corr_threshold = float(high_corr_threshold)
        self.high_vol_threshold = float(high_vol_threshold)
        self.cap_tighten_factor = float(cap_tighten_factor)
        self.recluster_seconds = int(recluster_minutes * 60)
        self.state_path = state_path

        self._last_refresh_ts: float = 0.0
        self._clusters: List[List[str]] = [[s] for s in self.symbols]
        self._regime: Dict[str, float | str] = {
            "avg_abs_corr": 0.0,
            "mean_vol": 0.0,
            "status": "normal",
            "cap_in_effect": self.base_cluster_cap,
        }

        self._ensure_dirs()
        self._load_state()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def maybe_refresh(self) -> None:
        """Refresh clusters/regime if stale."""
        if (time.time() - self._last_refresh_ts) < self.recluster_seconds:
            return
        try:
            prices = self._fetch_prices(self.symbols, self.lookback_days)
            if prices is None or prices.empty:
                return
            clusters, regime = self._compute_clusters_and_regime(prices)
            if clusters:
                self._clusters = clusters
                self._regime = regime
                self._last_refresh_ts = time.time()
                self._save_state()
        except Exception:
            # Fail safe: keep last good state
            pass

    def apply_cluster_caps(self, target_weights: Dict[str, float], redistribute: bool = True) -> Dict[str, float]:
        """
        Enforce per-cluster caps on a dictionary of symbol->target_weight.

        Args:
            target_weights: dict like {'AAPL': 0.12, 'MSFT': 0.10, ...}
            redistribute: if True, redistribute excess to symbols with capacity
        """
        self.maybe_refresh()

        # Normalize negatives/positives separately to preserve net exposure shape
        # but protect caps based on absolute long weight within cluster.
        # For simplicity, we cap by total *long* weight per cluster.
        tw = {s: float(target_weights.get(s, 0.0)) for s in self.symbols}
        long_weights = {s: max(0.0, w) for s, w in tw.items()}
        short_weights = {s: min(0.0, w) for s, w in tw.items()}  # currently not capped cluster-wise

        # Determine cap in effect based on regime
        cap = self._regime.get("cap_in_effect", self.base_cluster_cap)
        cap = float(cap)

        # First pass: cap long weights per cluster
        adjusted_long = long_weights.copy()
        excess = 0.0

        for cluster in self._clusters:
            cl_sum = sum(adjusted_long.get(s, 0.0) for s in cluster)
            if cl_sum > cap:
                scale = cap / (cl_sum + 1e-12)
                for s in cluster:
                    adjusted_long[s] *= scale
                excess += (cl_sum - cap)

        # Optional redistribution of excess to symbols with capacity
        if redistribute and excess > 1e-12:
            total_capacity = 0.0
            capacity: Dict[str, float] = {}
            for cluster in self._clusters:
                cl_sum = sum(adjusted_long.get(s, 0.0) for s in cluster)
                room = max(0.0, cap - cl_sum)
                # split capacity inside cluster across constituents proportional to current weights+epsilon
                eps = 1e-9
                cl_total = sum(adjusted_long.get(s, 0.0) + eps for s in cluster)
                for s in cluster:
                    share = (adjusted_long.get(s, 0.0) + eps) / cl_total
                    c = share * room
                    capacity[s] = capacity.get(s, 0.0) + c
                    total_capacity += c

            if total_capacity > 1e-12:
                factor = min(1.0, excess / total_capacity)
                for s in capacity:
                    adjusted_long[s] += capacity[s] * factor
                excess -= total_capacity * factor  # leftover (if capacity saturated)

        # Merge back long+short (we don't cap shorts here; can add if desired)
        out = {}
        for s in self.symbols:
            out[s] = adjusted_long.get(s, 0.0) + short_weights.get(s, 0.0)

        # Small cleanup: numerical drift
        for s in out:
            if abs(out[s]) < 1e-8:
                out[s] = 0.0

        return out

    def get_clusters(self) -> List[List[str]]:
        self.maybe_refresh()
        return [list(c) for c in self._clusters]

    def get_regime(self) -> Dict[str, float | str]:
        self.maybe_refresh()
        return dict(self._regime)

    # ------------------------------------------------------------
    # Internals: clustering + regime detection
    # ------------------------------------------------------------
    def _compute_clusters_and_regime(self, prices: pd.DataFrame) -> Tuple[List[List[str]], Dict[str, float | str]]:
        # Require at least a handful of rows
        if prices.shape[0] < 10:
            return [[s] for s in self.symbols], {
                "avg_abs_corr": 0.0,
                "mean_vol": 0.0,
                "status": "normal",
                "cap_in_effect": self.base_cluster_cap,
            }

        returns = prices.pct_change().dropna(how="any")
        corr = returns.corr().fillna(0.0)

        # Build adjacency with threshold on absolute correlation
        adj: Dict[str, set] = {s: set() for s in corr.columns}
        for i, s1 in enumerate(corr.columns):
            for j, s2 in enumerate(corr.columns):
                if j <= i:
                    continue
                if abs(corr.loc[s1, s2]) >= self.corr_threshold:
                    adj[s1].add(s2)
                    adj[s2].add(s1)

        clusters = self._connected_components(adj)

        # Regime detection
        avg_abs_corr = float(np.mean(np.abs(corr.values[np.triu_indices_from(corr, k=1)])))
        # cross-sectional mean of per-symbol daily std
        per_symbol_vol = returns.std().values
        mean_vol = float(np.mean(per_symbol_vol))

        high_corr = avg_abs_corr >= self.high_corr_threshold
        high_vol = mean_vol >= self.high_vol_threshold
        status = "high_corr_high_vol" if (high_corr and high_vol) else ("high_corr" if high_corr else ("high_vol" if high_vol else "normal"))

        cap_in_effect = self.base_cluster_cap * (self.cap_tighten_factor if status != "normal" else 1.0)

        regime = {
            "avg_abs_corr": round(avg_abs_corr, 6),
            "mean_vol": round(mean_vol, 6),
            "status": status,
            "cap_in_effect": cap_in_effect,
        }

        return clusters, regime

    @staticmethod
    def _connected_components(adj: Dict[str, set]) -> List[List[str]]:
        """Simple DFS connected components on adjacency dict."""
        seen = set()
        comps: List[List[str]] = []

        def dfs(node: str, comp: List[str]):
            seen.add(node)
            comp.append(node)
            for nb in adj.get(node, []):
                if nb not in seen:
                    dfs(nb, comp)

        for node in adj.keys():
            if node not in seen:
                comp: List[str] = []
                dfs(node, comp)
                comps.append(sorted(comp))

        # in case isolated nodes not present in adj (shouldn't happen), ensure coverage
        all_nodes = set(adj.keys())
        for node in all_nodes - set(sum(comps, [])):
            comps.append([node])

        return comps

    # ------------------------------------------------------------
    # Internals: price fetching
    # ------------------------------------------------------------
    def _fetch_prices(self, symbols: List[str], lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Try project data util first; fall back to yfinance if available.
        Must return a DataFrame indexed by date with columns per symbol (close prices).
        """
        # 1) Attempt project util
        try:
            from utils.market_data import get_price_history  # your project helper signature: (symbol, days) -> pd.Series
            series_list = []
            for s in symbols:
                ser = get_price_history(s, days=lookback_days)
                if ser is None or len(ser) == 0:
                    continue
                ser = ser.rename(s).astype(float)
                series_list.append(ser)
            if series_list:
                df = pd.concat(series_list, axis=1).dropna(how="all")
                return df
        except Exception:
            pass

        # 2) Fallback to yfinance if available
        try:
            import yfinance as yf
            end = pd.Timestamp.utcnow().normalize()
            start = end - pd.Timedelta(days=lookback_days + 10)
            data = yf.download(symbols, start=start.date(), end=end.date(), interval="1d", auto_adjust=True, progress=False, group_by="ticker")
            # yfinance returns multi-index if multiple symbols
            if isinstance(data.columns, pd.MultiIndex):
                closes = {}
                for s in symbols:
                    try:
                        closes[s] = data[(s, "Close")].astype(float)
                    except Exception:
                        continue
                if closes:
                    df = pd.DataFrame(closes).dropna(how="all")
                    return df
            else:
                # single symbol case
                df = pd.DataFrame({symbols[0]: data["Close"].astype(float)}).dropna(how="all")
                return df
        except Exception:
            pass

        return None

    # ------------------------------------------------------------
    # Persistence (optional)
    # ------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        d = os.path.dirname(self.state_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _save_state(self) -> None:
        try:
            blob = {
                "last_refresh_ts": self._last_refresh_ts,
                "clusters": self._clusters,
                "regime": self._regime,
                "params": {
                    "lookback_days": self.lookback_days,
                    "corr_threshold": self.corr_threshold,
                    "base_cluster_cap": self.base_cluster_cap,
                    "high_corr_threshold": self.high_corr_threshold,
                    "high_vol_threshold": self.high_vol_threshold,
                    "cap_tighten_factor": self.cap_tighten_factor,
                },
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            self._last_refresh_ts = float(blob.get("last_refresh_ts", 0.0))
            self._clusters = [list(map(str.upper, c)) for c in blob.get("clusters", self._clusters)]
            r = blob.get("regime", {})
            if isinstance(r, dict):
                self._regime.update(r)
        except Exception:
            pass
