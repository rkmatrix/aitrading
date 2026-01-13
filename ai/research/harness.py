# ai/research/harness.py
# -------------------------------------------------------------------
# Phase 10 — Research Harness
# - Unified walk-forward backtest with nested CV hyper-parameter selection
# - Optional Optuna optimization (falls back to grid search if unavailable)
# - Produces fold-by-fold metrics, trades, and consolidated KPIs
#
# Author: AITradeBot

from __future__ import annotations
import os
import json
import math
import time
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from utils.logger import log
from ai.data.price_resolver import PriceResolver
from ai.strategies.signal_engine import SignalEngine
from ai.allocators.portfolio_brain import PortfolioBrain
from ai.meta.regime_adapter import RegimeAdapter, RegimeConfig
from ai.allocators.alloc_optimizer import SignalBlender, CostModel, OrderSizer, AllocOptimizer

# Optional: Optuna for hyperopt
try:
    import optuna  # type: ignore
    _OPTUNA_OK = True
except Exception:
    _OPTUNA_OK = False


# ------------------------------
# Data classes & helpers
# ------------------------------
@dataclass
class ResearchConfig:
    symbols: List[str]
    period: str = "730d"
    interval: str = "1h"
    train_bars: int = 24*252   # ~1y of hourly bars (market hours proxy)
    test_bars: int = 24*21     # ~1 month
    step_bars: int = 24*21     # slide by test window
    max_folds: int = 18        # cap for speed
    results_dir: str = "data/research"
    use_optuna: bool = True
    n_trials: int = 40         # only used if optuna is available
    seed: int = 42


@dataclass
class ParamSpace:
    # Blender weights (will be projected to simplex)
    w_momentum: Tuple[float, float] = (0.1, 0.8)
    w_meanrev: Tuple[float, float] = (0.1, 0.8)
    w_macro: Tuple[float, float] = (0.0, 0.6)

    # Regime policy nudges (multipliers)
    mom_mult_volhi: Tuple[float, float] = (0.6, 1.2)
    mr_mult_volhi: Tuple[float, float] = (0.8, 1.4)

    # Global caps
    max_gross: Tuple[float, float] = (0.6, 1.05)
    per_pos_cap: Tuple[float, float] = (0.05, 0.20)


@dataclass
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe: float
    ret: float
    mdd: float
    turnover: float
    path_csv: str
    trades_csv: str
    params_json: str


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x) + 1e-12)
    return mu / sd


def _max_dd(eq: np.ndarray) -> float:
    peak = -1e12
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return abs(float(mdd))


def _project_simplex(a: np.ndarray) -> np.ndarray:
    a = np.maximum(a, 0)
    s = a.sum()
    return a / s if s > 0 else np.ones_like(a) / len(a)


def _grid(default_trials: int = 24) -> List[Dict[str,float]]:
    # deterministic mini-grid if Optuna is unavailable
    vals = []
    for wm in (0.2, 0.5, 0.7):
        for wr in (0.2, 0.4, 0.6):
            for wma in (0.0, 0.2, 0.4):
                for mg in (0.8, 1.0):
                    for pc in (0.10, 0.15, 0.20):
                        vals.append({
                            "w_momentum": wm,
                            "w_meanrev": wr,
                            "w_macro": wma,
                            "max_gross": mg,
                            "per_pos_cap": pc,
                            "mom_mult_volhi": 0.9,
                            "mr_mult_volhi": 1.1
                        })
    return vals[:default_trials]


class ResearchHarness:
    """
    Orchestrates:
      1) Data fetch via PriceResolver
      2) Walk-forward split
      3) Nested-CV selection of blender/regime caps per fold
      4) Simulation with PortfolioBrain (regime-aware constraints, but *no* executor)
      5) Persist metrics + trades; hand off to report generator
    """

    def __init__(self, cfg: ResearchConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        np.random.seed(self.cfg.seed)

        self.notifier = None
        self.resolver = PriceResolver(notifier=self.notifier)
        self.signal_engine = SignalEngine(cfg.symbols)

        regime_conf = RegimeConfig(
            n_regimes=3,
            memory_csv_path=os.path.join(self.cfg.results_dir, "regime_memory.csv"),
            enable_telegram_alerts=False
        )
        self.regime_adapter = RegimeAdapter(config=regime_conf, notifier=self.notifier)

        self.optimizer = AllocOptimizer()            # unused in backtest weight calc; retained for parity
        self.blender = SignalBlender()
        self.cost_model = CostModel()                # not used in research returns; sizing is target weights
        self.order_sizer = OrderSizer(self.cost_model)

        self.run_id = time.strftime("RH_%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.cfg.results_dir, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------------------
    # Public entry
    # -------------------------------
    def run(self) -> Dict:
        log(f"📚 Research run started → {self.out_dir}")
        close = self.resolver.get_close_matrix(self.cfg.symbols, self.cfg.period, self.cfg.interval)
        if close.empty:
            raise RuntimeError("Close matrix is empty; check data layer or app_config.")

        # compute returns per symbol
        rets = close.pct_change().dropna()

        folds = self._make_folds(rets.index)
        fold_results: List[FoldResult] = []

        for fidx, (tr_idx, te_idx) in enumerate(folds[: self.cfg.max_folds], start=1):
            tr_slice = slice(tr_idx[0], tr_idx[-1])
            te_slice = slice(te_idx[0], te_idx[-1])

            # Signals computed on entire window but sliced below
            momentum_all = self.signal_engine.momentum(close)
            meanrev_all = self.signal_engine.mean_reversion(close)
            macro_all   = self.signal_engine.macro_features(close)

            # Nested CV to choose params on training window
            best_params = self._select_params(
                close.loc[tr_slice],
                momentum_all.loc[tr_slice],
                meanrev_all.loc[tr_slice],
                macro_all.loc[tr_slice]
            )

            # Simulate on test window using the chosen params
            kpi, path_csv, trades_csv = self._simulate_fold(
                close.loc[tr_slice], close.loc[te_slice],
                momentum_all.loc[tr_slice], momentum_all.loc[te_slice],
                meanrev_all.loc[tr_slice], meanrev_all.loc[te_slice],
                macro_all.loc[tr_slice], macro_all.loc[te_slice],
                best_params,
                fold_id=fidx
            )

            fr = FoldResult(
                fold_id=fidx,
                train_start=str(tr_idx[0]), train_end=str(tr_idx[-1]),
                test_start=str(te_idx[0]), test_end=str(te_idx[-1]),
                sharpe=kpi["sharpe"], ret=kpi["ret"], mdd=kpi["mdd"], turnover=kpi["turnover"],
                path_csv=path_csv, trades_csv=trades_csv,
                params_json=json.dumps(best_params)
            )
            fold_results.append(fr)
            log(f"✅ Fold {fidx} | Sharpe={fr.sharpe:.2f} Ret={fr.ret:.2%} MDD={fr.mdd:.2%}")

        # Consolidate
        res_df = pd.DataFrame([asdict(x) for x in fold_results])
        res_path = os.path.join(self.out_dir, "fold_results.csv")
        res_df.to_csv(res_path, index=False)

        summary = {
            "run_id": self.run_id,
            "results_dir": self.out_dir,
            "symbols": self.cfg.symbols,
            "period": self.cfg.period,
            "interval": self.cfg.interval,
            "folds": len(res_df),
            "avg_sharpe": float(res_df["sharpe"].mean()) if not res_df.empty else 0.0,
            "avg_ret": float(res_df["ret"].mean()) if not res_df.empty else 0.0,
            "avg_mdd": float(res_df["mdd"].mean()) if not res_df.empty else 0.0,
            "avg_turnover": float(res_df["turnover"].mean()) if not res_df.empty else 0.0,
            "fold_results_csv": res_path
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        log(f"📦 Research run complete. Summary saved to {self.out_dir}")
        return summary

    # -------------------------------
    # Folds & selection
    # -------------------------------
    def _make_folds(self, idx: pd.Index) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
        idx = list(idx)
        folds = []
        i = self.cfg.train_bars
        while i + self.cfg.test_bars <= len(idx):
            tr = idx[i - self.cfg.train_bars: i]
            te = idx[i: i + self.cfg.test_bars]
            folds.append((tr, te))
            i += self.cfg.step_bars
        return folds

    def _select_params(
        self,
        close_tr: pd.DataFrame,
        mom_tr: pd.DataFrame,
        mr_tr: pd.DataFrame,
        macro_tr: pd.DataFrame
    ) -> Dict[str, float]:
        # Build a simple CV: split training into K folds by time
        K = 3
        T = len(close_tr)
        if T < (K + 1) * 50:
            # If too small, just return a safe default
            return {
                "w_momentum": 0.5, "w_meanrev": 0.3, "w_macro": 0.2,
                "max_gross": 1.0, "per_pos_cap": 0.15,
                "mom_mult_volhi": 0.8, "mr_mult_volhi": 1.2
            }

        splits = []
        step = T // (K + 1)
        for k in range(K):
            tr = close_tr.index[: step * (k + 1)]
            va = close_tr.index[step * (k + 1): step * (k + 2)]
            if len(va) > 10:
                splits.append((tr, va))

        def objective(params: Dict[str, float]) -> float:
            # returns negative loss (we want to maximize Sharpe)
            scores = []
            for tr_idx, va_idx in splits:
                score = self._cv_score(
                    close_tr.loc[tr_idx], close_tr.loc[va_idx],
                    mom_tr.loc[tr_idx], mom_tr.loc[va_idx],
                    mr_tr.loc[tr_idx], mr_tr.loc[va_idx],
                    macro_tr.loc[tr_idx], macro_tr.loc[va_idx],
                    params
                )
                scores.append(score)
            return float(np.mean(scores)) if scores else -1.0

        if self.cfg.use_optuna and _OPTUNA_OK:
            def _optuna_obj(trial):
                params = {
                    "w_momentum": trial.suggest_float("w_momentum", 0.1, 0.8),
                    "w_meanrev": trial.suggest_float("w_meanrev", 0.1, 0.8),
                    "w_macro": trial.suggest_float("w_macro", 0.0, 0.6),
                    "max_gross": trial.suggest_float("max_gross", 0.6, 1.05),
                    "per_pos_cap": trial.suggest_float("per_pos_cap", 0.05, 0.20),
                    "mom_mult_volhi": trial.suggest_float("mom_mult_volhi", 0.6, 1.2),
                    "mr_mult_volhi": trial.suggest_float("mr_mult_volhi", 0.8, 1.4),
                }
                return objective(params)

            study = optuna.create_study(direction="maximize")
            study.optimize(_optuna_obj, n_trials=self.cfg.n_trials, show_progress_bar=False)
            best = study.best_params
        else:
            best = None
            best_val = -1e9
            for params in _grid():
                val = objective(params)
                if val > best_val:
                    best = params
                    best_val = val

        # project blender weights to simplex
        w = np.array([best["w_momentum"], best["w_meanrev"], best["w_macro"]], dtype=float)
        w = _project_simplex(w)
        best["w_momentum"], best["w_meanrev"], best["w_macro"] = float(w[0]), float(w[1]), float(w[2])
        return best

    def _cv_score(
        self,
        close_tr: pd.DataFrame, close_va: pd.DataFrame,
        mom_tr: pd.DataFrame, mom_va: pd.DataFrame,
        mr_tr: pd.DataFrame, mr_va: pd.DataFrame,
        macro_tr: pd.DataFrame, macro_va: pd.DataFrame,
        params: Dict[str, float]
    ) -> float:
        # tiny simulation on validation
        weights_hist, rets = self._simulate_period(
            close_tr, close_va, mom_tr, mom_va, mr_tr, mr_va, macro_tr, macro_va, params
        )
        if rets.empty:
            return -1.0
        return _sharpe(rets.values)

    # -------------------------------
    # Simulation
    # -------------------------------
    def _simulate_fold(
        self,
        close_tr: pd.DataFrame, close_te: pd.DataFrame,
        mom_tr: pd.DataFrame, mom_te: pd.DataFrame,
        mr_tr: pd.DataFrame, mr_te: pd.DataFrame,
        macro_tr: pd.DataFrame, macro_te: pd.DataFrame,
        params: Dict[str, float],
        fold_id: int
    ) -> Tuple[Dict[str, float], str, str]:
        weights_hist, port_rets = self._simulate_period(
            close_tr, close_te, mom_tr, mom_te, mr_tr, mr_te, macro_tr, macro_te, params
        )
        if port_rets.empty:
            kpi = {"sharpe": 0.0, "ret": 0.0, "mdd": 0.0, "turnover": 0.0}
        else:
            eq = port_rets.cumsum()
            kpi = {
                "sharpe": _sharpe(port_rets.values),
                "ret": float(port_rets.mean()),
                "mdd": _max_dd(eq.values),
                "turnover": float(np.abs(weights_hist.diff().fillna(0)).sum(axis=1).mean()),
            }

        path_csv = os.path.join(self.out_dir, f"fold{fold_id:02d}_equity.csv")
        trades_csv = os.path.join(self.out_dir, f"fold{fold_id:02d}_weights.csv")
        pd.DataFrame({"equity": eq}, index=port_rets.index).to_csv(path_csv)
        weights_hist.to_csv(trades_csv)
        # save picked params
        with open(os.path.join(self.out_dir, f"fold{fold_id:02d}_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        return kpi, path_csv, trades_csv

    def _simulate_period(
        self,
        close_tr: pd.DataFrame, close_te: pd.DataFrame,
        mom_tr: pd.DataFrame, mom_te: pd.DataFrame,
        mr_tr: pd.DataFrame, mr_te: pd.DataFrame,
        macro_tr: pd.DataFrame, macro_te: pd.DataFrame,
        params: Dict[str, float]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Instantiate a lightweight PortfolioBrain with chosen caps; reuse regime adapter inside
        class _ShimBrain(PortfolioBrain):
            pass

        brain = _ShimBrain(
            symbols=self.cfg.symbols,
            max_gross_exposure=float(params["max_gross"]),
            turnover_limit=0.999,               # not enforced here
            regime_adapter=self.regime_adapter,
            notifier=None,
            optimizer=self.optimizer,
        )

        # override regime policy multipliers for high-vol regime (id=2), if present
        try:
            pol = brain.regime_adapter.policy_map.get(2, None)
            if pol:
                pol.feature_multipliers["momentum"] = float(params["mom_mult_volhi"])
                pol.feature_multipliers["mean_reversion"] = float(params["mr_mult_volhi"])
        except Exception:
            pass

        # Returns for test period
        rets_te = close_te.pct_change().dropna()
        weights_hist = pd.DataFrame(index=rets_te.index, columns=self.cfg.symbols, dtype=float)

        # Historical “basket” series for CV-like blender help, if used inside PortfolioBrain
        hist_signals = pd.DataFrame(columns=["momentum","meanrev","macro"])

        # walk forward through test bars, re-planning each bar using only past data
        prev_w = pd.Series(0.0, index=self.cfg.symbols)
        for t in rets_te.index:
            # training context up to t-1
            tr_ctx_end = close_tr.index[-1]
            tr_ctx = close_tr.loc[:tr_ctx_end]
            te_ctx = close_te.loc[:t].iloc[:-1]  # up to t-1
            ctx = pd.concat([tr_ctx, te_ctx], axis=0)

            # signals to time t-1
            mom_ctx = pd.concat([mom_tr, mom_te.loc[:t].iloc[:-1]], axis=0).reindex(ctx.index).dropna()
            mr_ctx  = pd.concat([mr_tr, mr_te.loc[:t].iloc[:-1]], axis=0).reindex(ctx.index).dropna()
            mac_ctx = pd.concat([macro_tr, macro_te.loc[:t].iloc[:-1]], axis=0).reindex(ctx.index).fillna(0.0)

            # current signals at t-1 for each symbol (last row)
            if len(mom_ctx) == 0 or len(mr_ctx) == 0:
                continue
            mom_now = mom_ctx.iloc[-1]
            mr_now  = mr_ctx .iloc[-1]
            mac_now = mac_ctx.iloc[-1] if len(mac_ctx)>0 else pd.Series(0.0, index=self.cfg.symbols)

            # set blender weights directly from selected params
            raw = (
                mom_now * params["w_momentum"] +
                mr_now  * params["w_meanrev"]  +
                mac_now * params["w_macro"]
            )
            gross = float(np.sum(np.abs(raw.values)))
            if gross > 0:
                raw = raw / gross

            # regime-aware constraints using SPY proxy
            try:
                # Build a proxy OHLCV for regime detection: use close from ctx and synthesize Volume if missing
                proxy = pd.DataFrame(index=ctx.index)
                proxy["Close"] = close_tr["SPY"] if "SPY" in close_tr.columns else ctx.mean(axis=1)
                proxy["Volume"] = 1_000_000.0
            except Exception:
                proxy = pd.DataFrame({"Close": ctx.mean(axis=1), "Volume": 1_000_000.0}, index=ctx.index)

            feature_snapshot = {
                "momentum": float(mom_now.mean()),
                "mean_reversion": float(mr_now.mean()),
                "macro_regime": float(mac_now.mean()) if isinstance(mac_now, pd.Series) else 0.0,
            }

            w = brain.compute_and_constrain_weights(
                raw_weights=raw,
                latest_df=proxy,
                feature_snapshot=feature_snapshot
            )

            # (Research path) skip cost-aware sizing & execution microstructure; record target weights
            weights_hist.loc[t] = w.reindex(self.cfg.symbols).fillna(0.0)

        # portfolio returns = sum(weights_{t-1} * returns_t)
        W = weights_hist.shift(1).fillna(0.0)
        port = (W * rets_te).sum(axis=1)
        return weights_hist.fillna(0.0), port
