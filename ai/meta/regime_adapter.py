# ai/meta/regime_adapter.py
# Phase 9.1 — Regime-Aware Meta-Learner
# ------------------------------------------------------------
# Detects market regimes using simple, robust features (volatility + liquidity),
# clusters regimes (KMeans by default; optional HMM if available), and enforces
# per-regime portfolio caps + feature-weight multipliers.
#
# Persists per-regime snapshots to data/meta/regime_memory.csv so you can audit
# behavior and evolve policies over time.
#
# Safe: no heavy dependencies; relies on numpy/pandas/sklearn only (sklearn is
# already in your project per earlier phases).
#
# Author: AITradeBot

from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception as e:
    raise ImportError("scikit-learn is required for RegimeAdapter (pip install scikit-learn)") from e


@dataclass
class RegimePolicy:
    """
    Policy knobs that change by regime. You can tune these live without retraining:
      - risk caps: total exposure, per-position caps, leverage, stop multiples
      - feature multipliers: emphasize/de-emphasize signals by regime
    """
    # risk
    max_gross_exposure: float              # e.g., 0.9 means 90% gross
    per_position_cap: float                # e.g., 0.15 means max 15% per name
    leverage_limit: float                  # e.g., 1.0 (no leverage), 1.5, etc.
    stop_atr_mult: float                   # e.g., 3.0 ATR
    # signals
    feature_multipliers: Dict[str, float]  # { "mom_20": 1.2, "meanrev_z": 0.8, ... }


@dataclass
class RegimeConfig:
    n_regimes: int = 3
    names: Optional[Dict[int, str]] = field(default_factory=lambda: {
        0: "Calm-HighLiq",
        1: "Normal",
        2: "HighVol-LowLiq",
    })
    # clustering
    random_state: int = 42
    use_hmm_if_available: bool = False      # reserved: KMeans is default
    # features
    vol_window: int = 20
    liq_window: int = 20
    min_history_bars: int = 200            # wait until we have enough bars
    # persistence + alerts
    memory_csv_path: str = "data/meta/regime_memory.csv"
    ensure_dirs: bool = True
    # telemetry (optional: will no-op if your notifier is absent)
    enable_telegram_alerts: bool = True
    telegram_channel: Optional[str] = None  # use your default if None


class RegimeAdapter:
    """
    - Computes rolling volatility & liquidity features.
    - Clusters into regimes with KMeans.
    - Applies regime-specific caps & signal multipliers to outgoing weights.
    - Persists every decision row to data/meta/regime_memory.csv.
    """
    def __init__(
        self,
        policy_map: Optional[Dict[int, RegimePolicy]] = None,
        config: Optional[RegimeConfig] = None,
        notifier: Optional[Any] = None,  # expects .notify(text, chat_id=None)
    ):
        self.config = config or RegimeConfig()
        self.notifier = notifier
        self._kmeans: Optional[KMeans] = None
        self._last_regime: Optional[int] = None

        # Default policy per regime (tune freely)
        self.policy_map = policy_map or {
            0: RegimePolicy(  # Calm-HighLiq
                max_gross_exposure=0.95,
                per_position_cap=0.18,
                leverage_limit=1.2,
                stop_atr_mult=3.5,
                feature_multipliers={
                    "momentum": 1.2,
                    "mean_reversion": 0.8,
                    "macro_regime": 1.0,
                },
            ),
            1: RegimePolicy(  # Normal
                max_gross_exposure=0.90,
                per_position_cap=0.15,
                leverage_limit=1.0,
                stop_atr_mult=3.0,
                feature_multipliers={
                    "momentum": 1.0,
                    "mean_reversion": 1.0,
                    "macro_regime": 1.0,
                },
            ),
            2: RegimePolicy(  # HighVol-LowLiq
                max_gross_exposure=0.70,
                per_position_cap=0.10,
                leverage_limit=0.9,
                stop_atr_mult=2.2,
                feature_multipliers={
                    "momentum": 0.8,
                    "mean_reversion": 1.2,
                    "macro_regime": 1.1,
                },
            ),
        }

        if self.config.ensure_dirs:
            os.makedirs(os.path.dirname(self.config.memory_csv_path), exist_ok=True)

        # Create header if file missing
        if not os.path.exists(self.config.memory_csv_path):
            pd.DataFrame(columns=[
                "timestamp",
                "regime_id",
                "regime_name",
                "vol20",
                "liq20",
                "gross_before",
                "gross_after",
                "leverage_limit",
                "per_position_cap",
                "stop_atr_mult",
                "feature_multipliers_json",
                "weights_json",
            ]).to_csv(self.config.memory_csv_path, index=False)

    # --------------------------- Public API --------------------------- #

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit KMeans on historical regime features.
        df must contain at least: ['Close', 'Volume'] indexed by datetime (ascending).
        """
        feats = self._compute_regime_features(df)
        if len(feats) < self.config.min_history_bars:
            raise ValueError(
                f"Not enough history to fit regimes "
                f"({len(feats)} < {self.app_config.min_history_bars})."
            )

        X = self._standardize(feats[["vol20", "inv_liq20"]].values)
        self._kmeans = KMeans(
            n_clusters=self.config.n_regimes,
            n_init=10,
            random_state=self.config.random_state,
        ).fit(X)

        # initialize last regime from last row
        last_regime = int(self._kmeans.predict(X[-1:].reshape(1, -1))[0])
        self._last_regime = last_regime

    def update_and_get_regime(
        self,
        df_tail: pd.DataFrame,
        allow_refit: bool = False,
    ) -> Tuple[int, str, Dict[str, float]]:
        """
        Update rolling features on the latest data and return (regime_id, name, metrics).
        If model isn't fit yet and we have enough data, auto-fit.
        """
        feats = self._compute_regime_features(df_tail)
        if self._kmeans is None:
            if len(feats) >= self.config.min_history_bars:
                self.fit(df_tail)
            else:
                # bootstrapped "Normal" until we can fit
                return 1, self.config.names.get(1, "Normal"), {
                    "vol20": float(feats["vol20"].iloc[-1]) if len(feats) else np.nan,
                    "liq20": float(feats["liq20"].iloc[-1]) if len(feats) else np.nan,
                }

        X_last = self._standardize(feats[["vol20", "inv_liq20"]].values)[-1:]
        regime_id = int(self._kmeans.predict(X_last)[0])
        regime_name = self.config.names.get(regime_id, f"Regime{regime_id}")

        # Optional periodic refit to avoid drift (off by default)
        if allow_refit and len(feats) >= self.config.min_history_bars:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kmeans = KMeans(
                    n_clusters=self.config.n_regimes,
                    n_init=10,
                    random_state=self.config.random_state,
                ).fit(self._standardize(feats[["vol20", "inv_liq20"]].values))

        # Alert on regime change
        if self._last_regime is not None and regime_id != self._last_regime:
            self._alert_regime_change(self._last_regime, regime_id, feats.iloc[-1])
        self._last_regime = regime_id

        return regime_id, regime_name, {
            "vol20": float(feats["vol20"].iloc[-1]),
            "liq20": float(feats["liq20"].iloc[-1]),
        }

    def apply_overrides(
        self,
        raw_weights: pd.Series,
        regime_id: int,
        metrics: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None,
        feature_snapshot: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Scale weights and clip by per-regime caps. Log the decision to CSV.
        - raw_weights: pd.Series indexed by symbol, summing to <= 1 by absolute (gross)
        - feature_snapshot: optional dict of canonical feature scores (e.g., {'momentum':0.6,...})
        """
        pol = self._get_policy(regime_id)

        w = raw_weights.copy().astype(float).fillna(0.0)
        gross_before = float(np.sum(np.abs(w.values)))

        # Feature multipliers → scale weights proportionally to blended signal “strength”
        if feature_snapshot and pol.feature_multipliers:
            # compute an overall multiplier from provided features
            m = 0.0
            denom = 0.0
            for name, val in feature_snapshot.items():
                mult = pol.feature_multipliers.get(name, 1.0)
                m += float(val) * float(mult)
                denom += abs(float(val)) if val is not None else 0.0
            overall = (m / denom) if denom > 0 else 1.0
            w *= float(np.clip(overall, 0.6, 1.4))  # keep sane bounds

        # Per position cap
        cap = float(pol.per_position_cap)
        w = w.clip(lower=-cap, upper=cap)

        # Gross exposure cap
        gross = float(np.sum(np.abs(w.values)))
        if gross > pol.max_gross_exposure and gross > 0:
            w *= (pol.max_gross_exposure / gross)

        # Leverage hard limit (if portfolio uses margin elsewhere)
        # Here we interpret as another gross limiter
        gross_after = float(np.sum(np.abs(w.values)))
        if gross_after > pol.leverage_limit and gross_after > 0:
            w *= (pol.leverage_limit / gross_after)
            gross_after = float(np.sum(np.abs(w.values)))

        # Persist snapshot
        ts = (timestamp or pd.Timestamp.utcnow()).isoformat()
        self._append_memory(
            timestamp=ts,
            regime_id=regime_id,
            regime_name=self.config.names.get(regime_id, f"Regime{regime_id}"),
            vol20=metrics.get("vol20", np.nan),
            liq20=metrics.get("liq20", np.nan),
            gross_before=gross_before,
            gross_after=gross_after,
            leverage_limit=pol.leverage_limit,
            per_position_cap=pol.per_position_cap,
            stop_atr_mult=pol.stop_atr_mult,
            feature_multipliers=pol.feature_multipliers,
            weights=w.to_dict(),
        )

        return w

    def current_policy(self, regime_id: int) -> RegimePolicy:
        return self._get_policy(regime_id)

    # ----------------------- Internal helpers ------------------------ #

    def _get_policy(self, regime_id: int) -> RegimePolicy:
        # fallback to "Normal" policy if unknown
        return self.policy_map.get(regime_id, self.policy_map[1])

    def _compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with columns: ['vol20','liq20','inv_liq20'] (indexed by dt)
        - vol20: rolling realized vol of close-to-close returns (annualized)
        - liq20: rolling mean dollar volume (Close * Volume), log10-scaled
        """
        if "Close" not in df.columns or "Volume" not in df.columns:
            raise ValueError("DataFrame must contain ['Close','Volume'] columns.")

        df = df.sort_index()
        ret = df["Close"].pct_change()
        vol20 = ret.rolling(self.config.vol_window).std() * np.sqrt(252)  # annualized
        dollar_vol = (df["Close"] * df["Volume"]).rolling(self.config.liq_window).mean()
        liq20 = np.log10(dollar_vol.replace(0, np.nan))

        feats = pd.DataFrame({
            "vol20": vol20,
            "liq20": liq20,
        }).dropna()

        feats["inv_liq20"] = -feats["liq20"]  # high liquidity → lower inv_liq
        return feats

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0) + 1e-12
        return (X - mu) / sd

    def _append_memory(
        self,
        timestamp: str,
        regime_id: int,
        regime_name: str,
        vol20: float,
        liq20: float,
        gross_before: float,
        gross_after: float,
        leverage_limit: float,
        per_position_cap: float,
        stop_atr_mult: float,
        feature_multipliers: Dict[str, float],
        weights: Dict[str, float],
    ) -> None:
        row = {
            "timestamp": timestamp,
            "regime_id": regime_id,
            "regime_name": regime_name,
            "vol20": vol20,
            "liq20": liq20,
            "gross_before": gross_before,
            "gross_after": gross_after,
            "leverage_limit": leverage_limit,
            "per_position_cap": per_position_cap,
            "stop_atr_mult": stop_atr_mult,
            "feature_multipliers_json": json.dumps(feature_multipliers, separators=(",", ":"), sort_keys=True),
            "weights_json": json.dumps({k: float(v) for k, v in weights.items()}, separators=(",", ":"), sort_keys=True),
        }
        try:
            pd.DataFrame([row]).to_csv(
                self.config.memory_csv_path,
                mode="a",
                header=False,
                index=False
            )
        except Exception as e:
            warnings.warn(f"Failed to append regime memory: {e}")

    def _alert_regime_change(self, prev_id: int, new_id: int, last_feat_row: pd.Series) -> None:
        if not self.config.enable_telegram_alerts:
            return
        msg = (
            "⚠️ Regime change detected\n"
            f"Prev: {self.app_config.names.get(prev_id, prev_id)} → "
            f"Now: {self.app_config.names.get(new_id, new_id)}\n"
            f"vol20={last_feat_row.get('vol20'):.3f}, "
            f"liq20={last_feat_row.get('liq20'):.3f}"
        )
        # If your project has utils.notifier or similar, call it here
        try:
            if self.notifier is not None and hasattr(self.notifier, "notify"):
                self.notifier.notify(msg, chat_id=self.config.telegram_channel)
        except Exception:
            pass
