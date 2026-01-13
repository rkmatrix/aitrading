# ai/allocators/alloc_optimizer.py
# -------------------------------------------------------------------
# Phase 9.0 — Allocation Intelligence++
# 1) Online "Bayesian-like" tuning of penalty weights for:
#      - turnover
#      - drawdown
#      - concentration
#    Uses Thompson-sampling over a small, smart candidate grid
#    and persists state to JSON.
#
# 2) Dynamic signal ensembling:
#    Expanding-window cross-validation to choose blending weights
#    (ridge across {alpha} grid), projected to simplex (>=0, sum=1).
#
# 3) Cost-aware execution:
#    - Simple parametric slippage + fee model
#    - OrderSizer that throttles trade size vs expected alpha decay
#      and returns a "sized_weights" vector ready for execution.
#
# Dependency-light: numpy, pandas only (sklearn optional but not required).
#
# Author: AITradeBot

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------
# Utilities
# ---------------------
def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto probability simplex (non-negative, sum=1)."""
    if np.all(v <= 0):
        # fallback equal weights
        n = len(v)
        return np.ones(n) / n
    v = np.maximum(v, 0)
    s = v.sum()
    return v / s if s > 0 else v


def _safe_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 3:
        return 0.0
    mu = np.nanmean(returns)
    sd = np.nanstd(returns) + 1e-12
    return float(mu / sd)


def _max_drawdown(eq: np.ndarray) -> float:
    """Max drawdown on equity curve array."""
    if len(eq) == 0:
        return 0.0
    peak = -np.inf
    mdd = 0.0
    for x in eq:
        peak = max(peak, x)
        mdd = min(mdd, x - peak)
    return abs(float(mdd))


# ---------------------
# Signal Blender
# ---------------------
@dataclass
class BlenderConfig:
    alphas: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)  # ridge strengths
    min_samples: int = 60
    valid_tail: int = 20         # last N points as validation window
    decay: float = 0.97          # EMA decay for robustness


class SignalBlender:
    """
    Expanding-window CV to weight (momentum, meanrev, macro) streams.
    Uses ridge regression on historical signal-returns mapping.
    Projects to simplex to keep weights interpretable and stable.
    """

    def __init__(self, config: Optional[BlenderConfig] = None):
        self.cfg = config or BlenderConfig()
        self.last_weights = np.array([0.5, 0.3, 0.2], dtype=float)

    def fit_weights(
        self,
        signal_matrix: pd.DataFrame,  # columns: ['momentum','meanrev','macro'] per symbol avg or basket
        target_returns: pd.Series,    # realized portfolio return series aligned to rows
    ) -> np.ndarray:
        # Align and ensure numpy arrays
        df = pd.concat([signal_matrix, target_returns.rename("y")], axis=1).dropna()
        if len(df) < self.cfg.min_samples:
            return self.last_weights

        X = df[["momentum", "meanrev", "macro"]].values
        y = df["y"].values

        # expanding window: train on [0:-valid_tail], validate on tail
        t = len(df) - self.cfg.valid_tail
        if t <= 5:
            return self.last_weights

        Xtr, ytr = X[:t], y[:t]
        Xv, yv = X[t:], y[t:]

        best_sharpe = -1e9
        best_w = self.last_weights

        # closed-form ridge: w = (X^T X + alpha I)^(-1) X^T y
        XtX = Xtr.T @ Xtr
        Xty = Xtr.T @ ytr
        dim = XtX.shape[0]
        I = np.eye(dim)
        for a in self.cfg.alphas:
            w = np.linalg.solve(XtX + a * I, Xty)
            w = _project_to_simplex(w)
            # validate
            pred = Xv @ w
            s = _safe_sharpe(pred - yv)  # or use correlation between pred and yv
            # we want predictions to correlate with yv; use inverse error metric:
            s = float(np.corrcoef(pred, yv)[0, 1]) if np.std(pred) > 1e-12 else -1.0
            if np.isnan(s):
                s = -1.0
            if s > best_sharpe:
                best_sharpe = s
                best_w = w

        # smooth with EMA for stability
        self.last_weights = self.cfg.decay * self.last_weights + (1 - self.cfg.decay) * best_w
        self.last_weights = _project_to_simplex(self.last_weights)
        return self.last_weights.copy()


# ---------------------
# Cost Model & Order Sizer
# ---------------------
@dataclass
class CostModelConfig:
    fee_per_share: float = 0.0035           # $ fees per share
    slip_k: float = 0.0006                  # slippage coeff (fractional of notional)^0.5
    min_trade_notional: float = 25.0        # do not trade below this notional
    max_trade_fraction: float = 1.0         # cap per-iteration fraction of desired delta
    alpha_half_life_min: int = 240          # minutes to half-decay of alpha (4h)


class CostModel:
    """Simple parametric cost model with sqrt slippage + linear fees."""

    def __init__(self, cfg: Optional[CostModelConfig] = None):
        self.cfg = cfg or CostModelConfig()

    def estimate_cost_fraction(self, trade_frac: float) -> float:
        """
        Approximate cost as bps of notional for a given trade fraction.
        sqrt model: slip ≈ k * sqrt(f), fees linear in f (small).
        Return fraction (e.g., 0.0008 means 8 bps).
        """
        k = self.cfg.slip_k
        fees = 0.00005 * trade_frac  # 0.5 bps per 10% notional traded
        slip = k * np.sqrt(max(trade_frac, 1e-9))
        return float(slip + fees)


class OrderSizer:
    """
    Chooses a fraction of desired trade that balances cost vs alpha decay.
    We use a closed-form heuristic:
      trade_frac* = clip( sqrt( benefit / (C * volatility) ), 0, max_trade_fraction )
    Here, benefit is proxied by signal_strength, and C is related to slip_k.
    """

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def size(self, desired_weights: pd.Series, current_weights: pd.Series,
             signal_strength: float) -> pd.Series:
        delta = (desired_weights - current_weights).fillna(0.0)
        # Map signal strength (0..1+) to benefit; avoid zero
        benefit = max(signal_strength, 1e-3)
        # Translate cost param to an effective C
        C = max(self.cost_model.cfg.slip_k, 1e-6)
        frac = np.sqrt(benefit / (C * 10.0))           # heuristic scaling
        frac = float(np.clip(frac, 0.0, self.cost_model.cfg.max_trade_fraction))

        # If very small desired notional, skip (min_trade_notional handled upstream by executor)
        sized = current_weights + delta * frac
        # re-normalize to keep gross <= 1.05 before further constraints downstream
        gross = float(np.sum(np.abs(sized)))
        if gross > 1.05:
            sized *= (1.05 / gross)
        return sized


# ---------------------
# Online Penalty Optimizer (Thompson over candidate grid)
# ---------------------
@dataclass
class PenaltyCandidate:
    turnover: float
    drawdown: float
    concentration: float


@dataclass
class OptimizerState:
    candidates: List[PenaltyCandidate]
    mus: List[float]         # posterior mean utility per candidate
    sigmas: List[float]      # posterior std per candidate
    counts: List[int]        # how many updates per candidate
    last_idx: int            # last chosen index


class AllocOptimizer:
    """
    Maintains a grid of penalty triplets and uses Thompson Sampling to pick one.
    Reward = (Sharpe-like) - penalties*(metrics).
    This is intentionally simple & robust without external packages.
    """

    def __init__(self, state_path: str = "data/meta/alloc_optimizer_state.json"):
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        self.state_path = state_path
        self.state = self._load_or_init()

    # ---- public API ----
    def current_penalties(self) -> Dict[str, float]:
        c = self.state.candidates[self.state.last_idx]
        return {
            "turnover": c.turnover,
            "drawdown": c.drawdown,
            "concentration": c.concentration,
        }

    def select_next(self) -> Dict[str, float]:
        # Thompson sampling draw from Normal(mus, sigmas)
        draws = []
        for mu, sd in zip(self.state.mus, self.state.sigmas):
            sd = max(sd, 1e-3)
            draws.append(np.random.normal(mu, sd))
        idx = int(np.argmax(draws))
        self.state.last_idx = idx
        self._save()
        return self.current_penalties()

    def update_reward(self, realized: Dict[str, float]) -> None:
        """
        realized = {
          'ret': float,        # avg return over window
          'sharpe': float,
          'turnover': float,   # realized
          'drawdown': float,   # realized max DD over window
          'concentration': float, # e.g., Herfindahl index or max pos
        }
        Reward = sharpe - (lam_t*turnover + lam_d*drawdown + lam_c*concentration)
        """
        idx = self.state.last_idx
        lam = self.state.candidates[idx]
        reward = realized.get("sharpe", 0.0) \
                 - lam.turnover * realized.get("turnover", 0.0) \
                 - lam.drawdown * realized.get("drawdown", 0.0) \
                 - lam.concentration * realized.get("concentration", 0.0)

        # simple Bayesian normal-normal update (unknown variance → use Welford-ish)
        n = self.state.counts[idx]
        mu = self.state.mus[idx]
        sd = self.state.sigmas[idx]

        # update mean
        new_mu = (mu * n + reward) / (n + 1)
        # update std: shrink slowly
        new_sd = max(0.5 * sd, 0.02)

        self.state.mus[idx] = float(new_mu)
        self.state.sigmas[idx] = float(new_sd)
        self.state.counts[idx] = int(n + 1)
        self._save()

    # ---- helpers ----
    def _load_or_init(self) -> OptimizerState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    js = json.load(f)
                cands = [PenaltyCandidate(**c) for c in js["candidates"]]
                return OptimizerState(
                    candidates=cands,
                    mus=js["mus"],
                    sigmas=js["sigmas"],
                    counts=js["counts"],
                    last_idx=js["last_idx"],
                )
            except Exception:
                pass

        # default candidate grid
        grid = [
            PenaltyCandidate(0.1, 0.5, 0.2),
            PenaltyCandidate(0.2, 0.5, 0.3),
            PenaltyCandidate(0.1, 0.8, 0.2),
            PenaltyCandidate(0.3, 0.6, 0.5),
            PenaltyCandidate(0.05, 0.4, 0.15),
            PenaltyCandidate(0.15, 0.7, 0.25),
        ]
        k = len(grid)
        state = OptimizerState(
            candidates=grid,
            mus=[0.0] * k,
            sigmas=[0.5] * k,
            counts=[0] * k,
            last_idx=0,
        )
        self._save_obj(state)
        return state

    def _save(self) -> None:
        self._save_obj(self.state)

    def _save_obj(self, state: OptimizerState) -> None:
        with open(self.state_path, "w") as f:
            json.dump({
                "candidates": [asdict(c) for c in state.candidates],
                "mus": state.mus,
                "sigmas": state.sigmas,
                "counts": state.counts,
                "last_idx": state.last_idx,
            }, f, indent=2)
