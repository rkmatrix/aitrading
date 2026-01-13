# ai/envs/portfolio_env.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces  # type: ignore

from utils.logger import log
from ai.rewards.multi_objective import MultiObjectiveReward
from ai.allocators.portfolio_brain import PortfolioBrain
from ai.data.price_resolver import PriceResolver
from app_config import CFG

class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        symbols: list[str],
        period: str = "730d",
        interval: str = "1d",
        max_gross: float = 1.0,
        window: int = 60,
        cost_bps: float = 1.0,
        risk_penalty: float = 4.0,
        turnover_penalty: float = 2.0,
        cost_penalty: float = 1.0,
        seed: int = 42,
        use_brain_constraints: bool = True,
        # NEW: guardrails
        max_daily_loss: float = 0.05,      # 5% daily stop (soft end episode)
        max_pos_loss: float = 0.15,        # 15% per-position cumulative (soft cap)
    ):
        super().__init__()
        self.symbols = symbols
        self.n = len(symbols)
        self.period = period
        self.interval = interval
        self.max_gross = max_gross
        self.window = window
        self.rng = np.random.default_rng(seed)

        # Data
        self.resolver = PriceResolver()
        df = self.resolver.get_close_matrix(symbols, period, interval)
        if df.empty:
            raise RuntimeError("❌ No valid price data found for environment.")
        self.close = df
        self.rets = self.close.pct_change().fillna(0.0)
        log(f"✅ Loaded {len(self.close)} rows × {len(self.close.columns)} symbols for env")

        # Reward & constraints
        self.rewarder = MultiObjectiveReward(
            risk_penalty=risk_penalty,
            turnover_penalty=turnover_penalty,
            cost_penalty=cost_penalty,
            cost_bps=cost_bps,
        )
        self.use_brain_constraints = use_brain_constraints
        self.brain = PortfolioBrain(symbols, max_gross_exposure=max_gross) if use_brain_constraints else None

        # Guardrails
        self.max_daily_loss = float(max_daily_loss)
        self.max_pos_loss = float(max_pos_loss)

        # Spaces
        self.fdim = self.n * 3 + self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.fdim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

        self._reset_state()

    # ----------------------------- Helpers -----------------------------
    def _reset_state(self):
        self.idx = self.window
        self.prev_w = np.zeros(self.n, dtype=np.float32)
        self.equity = 1.0
        self.equity_path = [self.equity]
        self.done_flag = False
        self._day_start_equity = self.equity
        self.pos_pnl = np.zeros(self.n, dtype=np.float32)  # cumulative by symbol

    def _features(self, t: int) -> np.ndarray:
        # rolling stats with safe indexing
        r = self.rets[self.symbols]
        mom = r.rolling(20).mean().iloc[t - 1].values
        meanrev = -r.rolling(5).mean().iloc[t - 1].values
        vol = r.rolling(20).std().iloc[t - 1].values
        feat = np.concatenate([mom, meanrev, vol, self.prev_w], dtype=np.float32)
        feat = np.nan_to_num(feat, copy=False)
        return feat

    def _constrain(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, -1.0, 1.0)
        gross = np.sum(np.abs(w))
        if gross > 0:
            w *= self.max_gross / max(gross, 1e-8)
        if self.use_brain_constraints and self.brain is not None:
            try:
                proxy = pd.DataFrame(index=self.close.index[: self.idx])
                proxy["Close"] = self.close[self.symbols].iloc[: self.idx].mean(axis=1)
                proxy["Volume"] = 1_000_000.0
                series = pd.Series(w, index=self.symbols)
                series = self.brain.compute_and_constrain_weights(
                    raw_weights=series,
                    latest_df=proxy,
                    feature_snapshot={"momentum": float(np.mean(w)), "mean_reversion": float(-np.mean(w))},
                )
                w = series.reindex(self.symbols).fillna(0.0).values.astype(np.float32)
            except Exception as e:
                # Single log to avoid spam; keep silent afterwards
                log(f"⚠️ Brain constraint failed: {e}")
        return w

    # ------------------------------ Gym API ----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._features(self.idx), {}

    def step(self, action: np.ndarray):
        if self.done_flag:
            return self._features(self.idx - 1), 0.0, True, False, {}

        w = self._constrain(action)
        t = self.idx

        # realized return at t using weights placed at t-1
        r_vec = self.rets.iloc[t][self.symbols].values
        r_t = float(np.dot(r_vec, self.prev_w))

        # per-position pnl accumulation (for soft per-position guardrail)
        self.pos_pnl += (r_vec * self.prev_w).astype(np.float32)

        turnover = np.sum(np.abs(w - self.prev_w))
        cost = self.rewarder.cost_from_turnover(turnover)

        hist = self.rets.iloc[t - self.window + 1 : t + 1][self.symbols].values
        port_hist = (hist * self.prev_w.reshape(1, -1)).sum(axis=1)
        risk = np.std(port_hist) if len(port_hist) > 3 else 0.0

        reward = self.rewarder.compute(r_t, risk, turnover, cost)
        self.equity *= (1.0 + r_t - cost)
        self.equity_path.append(self.equity)
        self.prev_w = w
        self.idx += 1

        # --------- Circuit breakers (soft) ----------
        # Daily loss breaker
        # Reset "day start" when date changes (1d interval)
        if self.interval.endswith("1d"):
            if self.idx >= 2:
                prev_date = self.close.index[self.idx - 2].date()
                curr_date = self.close.index[self.idx - 1].date()
                if curr_date != prev_date:
                    self._day_start_equity = self.equity

            day_dd = (self._day_start_equity - self.equity) / max(self._day_start_equity, 1e-12)
            if day_dd >= self.max_daily_loss:
                # End episode softly
                self.done_flag = True

        # Per-position cumulative loss breaker (soft cap → shrink weights)
        too_bad = np.where(self.pos_pnl < -self.max_pos_loss)[0]
        if len(too_bad) > 0:
            self.prev_w[too_bad] = 0.0  # clamp next step
            self.pos_pnl[too_bad] = 0.0

        if self.idx >= len(self.close):
            self.done_flag = True

        obs = self._features(self.idx if not self.done_flag else self.idx - 1)
        info = {"r_t": r_t, "risk": risk, "turnover": turnover, "cost": cost, "equity": self.equity}
        return obs, float(reward), self.done_flag, False, info

    def render(self): ...
    def close(self): ...
