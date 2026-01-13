# ai/health/drift_monitor.py
# -------------------------------------------------------------------
# Phase 9.5 — Feature Drift Monitor
# - Tracks distributional drift for momentum / mean-reversion / macro signals
# - Uses PSI (Population Stability Index) and KS statistic
# - Persists rolling reference samples to CSV for reproducibility
#
# Author: AITradeBot

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

DEFAULT_REF_PATH = "data/meta/feature_ref_signals.csv"
DEFAULT_HEALTH_LOG = "data/logs/health.csv"


@dataclass
class DriftThresholds:
    # PSI: <0.1 no issue, 0.1–0.25 moderate, >0.25 severe (common industry guidance)
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    # KS: values near 0.1–0.2+ often indicate significant shift (rule-of-thumb)
    ks_warn: float = 0.10
    ks_alert: float = 0.20


class DriftMonitor:
    """
    Maintains and evaluates reference distributions for signals.
    Reference updates gradually (EMA-style) to avoid overreacting.
    """

    def __init__(
        self,
        ref_path: str = DEFAULT_REF_PATH,
        health_log_path: str = DEFAULT_HEALTH_LOG,
        max_ref_rows: int = 5000,
        thresholds: Optional[DriftThresholds] = None,
        ema_decay: float = 0.97,      # how slowly to fold in new reference stats
        min_ref_rows: int = 200,      # need at least this many rows before judging
    ):
        self.ref_path = ref_path
        self.health_log_path = health_log_path
        self.max_ref_rows = max_ref_rows
        self.th = thresholds or DriftThresholds()
        self.ema_decay = ema_decay
        self.min_ref_rows = min_ref_rows

        os.makedirs(os.path.dirname(self.ref_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.health_log_path), exist_ok=True)

        if not os.path.exists(self.health_log_path):
            pd.DataFrame(columns=[
                "timestamp", "type", "metric", "psi", "ks", "status", "details_json"
            ]).to_csv(self.health_log_path, index=False)

    # ------------------------ Public API ------------------------ #

    def evaluate_signals(
        self,
        momentum: pd.Series,
        meanrev: pd.Series,
        macro: Optional[pd.Series] = None,
        timestamp: Optional[pd.Timestamp] = None,
        write_ref_on_ok: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns dict per feature: {'psi': x, 'ks': y, 'status': 'ok|warn|alert'}
        Also appends a row to health log.
        """
        macro = macro if macro is not None else pd.Series(0.0, index=momentum.index)
        cur = pd.DataFrame({
            "momentum": momentum.fillna(0.0).values,
            "meanrev": meanrev.fillna(0.0).values,
            "macro": macro.fillna(0.0).values,
        })

        ref = self._load_ref()
        results: Dict[str, Dict[str, float]] = {}
        ts = (timestamp or pd.Timestamp.utcnow()).isoformat()

        if ref is None or len(ref) < self.min_ref_rows:
            # bootstrap reference
            self._bootstrap_ref(cur)
            self._log(ts, "drift", "bootstrap", psi=np.nan, ks=np.nan, status="bootstrap",
                      details={"rows": int(len(cur))})
            return {k: {"psi": np.nan, "ks": np.nan, "status": "bootstrap"} for k in cur.columns}

        for col in cur.columns:
            psi = self._psi(ref[col].values, cur[col].values)
            ks = self._ks(ref[col].values, cur[col].values)
            status = "ok"
            if psi >= self.th.psi_alert or ks >= self.th.ks_alert:
                status = "alert"
            elif psi >= self.th.psi_warn or ks >= self.th.ks_warn:
                status = "warn"
            results[col] = {"psi": float(psi), "ks": float(ks), "status": status}

        # log one row per evaluation with summary
        summary = {
            "momentum": results["momentum"],
            "meanrev": results["meanrev"],
            "macro": results["macro"],
        }
        worst = max((d["status"] for d in results.values()), key=lambda s: {"ok":0,"warn":1,"alert":2}[s])
        self._log(ts, "drift", "signals", psi=max(d["psi"] for d in results.values()),
                  ks=max(d["ks"] for d in results.values()), status=worst, details=summary)

        # refresh reference if not alert (slowly)
        if write_ref_on_ok and worst != "alert":
            self._update_ref(cur)

        return results

    # ------------------------ Internals ------------------------ #

    def _load_ref(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.ref_path):
            return None
        try:
            df = pd.read_csv(self.ref_path)
            return df[["momentum", "meanrev", "macro"]].dropna()
        except Exception:
            return None

    def _bootstrap_ref(self, cur: pd.DataFrame) -> None:
        # seed with current; clipped to max_ref_rows
        df = cur.copy()
        if len(df) > self.max_ref_rows:
            df = df.sample(self.max_ref_rows, random_state=42)
        df.to_csv(self.ref_path, index=False)

    def _update_ref(self, cur: pd.DataFrame) -> None:
        ref = self._load_ref()
        if ref is None or ref.empty:
            self._bootstrap_ref(cur)
            return
        # concatenate then downsample to max_ref_rows keeping recent bias
        df = pd.concat([ref, cur], axis=0, ignore_index=True)
        if len(df) > self.max_ref_rows:
            # keep last 80% recent + sample remaining from older
            keep_recent = int(self.max_ref_rows * 0.8)
            recent = df.iloc[-keep_recent:]
            older_needed = self.max_ref_rows - keep_recent
            older = df.iloc[:-keep_recent].sample(older_needed, random_state=42) if older_needed > 0 else df.iloc[0:0]
            df = pd.concat([older, recent], axis=0).reset_index(drop=True)
        df.to_csv(self.ref_path, index=False)

    @staticmethod
    def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        # bin edges from reference quantiles
        if len(ref) < 5 or len(cur) < 5:
            return 0.0
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(ref, qs)
        # guard against identical edges
        edges = np.unique(edges)
        if len(edges) <= 2:
            edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)

        r_hist, _ = np.histogram(ref, bins=edges)
        c_hist, _ = np.histogram(cur, bins=edges)

        r_pct = r_hist / max(r_hist.sum(), 1)
        c_pct = c_hist / max(c_hist.sum(), 1)

        # avoid zeros
        r_pct = np.where(r_pct == 0, 1e-6, r_pct)
        c_pct = np.where(c_pct == 0, 1e-6, c_pct)

        psi = np.sum((c_pct - r_pct) * np.log(c_pct / r_pct))
        return float(max(psi, 0.0))

    @staticmethod
    def _ks(ref: np.ndarray, cur: np.ndarray) -> float:
        # simple ECDF-based KS
        if len(ref) < 5 or len(cur) < 5:
            return 0.0
        ref = np.sort(ref)
        cur = np.sort(cur)
        all_vals = np.sort(np.unique(np.concatenate([ref, cur])))
        def ecdf(x, arr):
            # proportion <= x
            return np.searchsorted(arr, x, side="right") / len(arr)
        diffs = [abs(ecdf(v, ref) - ecdf(v, cur)) for v in all_vals]
        return float(max(diffs))

    def _log(self, ts: str, typ: str, metric: str, psi: float, ks: float, status: str, details: Dict):
        row = {
            "timestamp": ts,
            "type": typ,
            "metric": metric,
            "psi": psi,
            "ks": ks,
            "status": status,
            "details_json": json.dumps(details, separators=(",", ":"), sort_keys=True),
        }
        try:
            pd.DataFrame([row]).to_csv(self.health_log_path, mode="a", index=False, header=False)
        except Exception:
            pass
