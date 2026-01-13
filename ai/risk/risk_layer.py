# ai/risk/risk_layer.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import datetime as dt

META_DIR = Path("data") / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV = META_DIR / "performance_memory_trades.csv"
RISK_STATE_JSON = META_DIR / "risk_state.json"

SENT_PATH = Path("data/sentiment/latest_sentiment.csv")
MACRO_PATH = Path("data/macro/latest_macro.csv")


# ---------------------------------------------------------------------
# Data class for current risk state
# ---------------------------------------------------------------------
@dataclass
class RiskState:
    regime: str = "Normal"
    realized_vol_20d: float = 0.0
    max_drawdown_60d: float = 0.0
    avg_sentiment: float = 0.0
    macro_stress: float = 0.0
    leverage_mult: float = 1.0
    maxpos_mult: float = 1.0
    cash_floor_add: float = 0.00
    circuit_breaker: bool = False
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    order_intensity: float = 1.0
    cooldown_active: bool = False
    cooldown_until_iso: str = ""


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_state(state: RiskState) -> None:
    with open(RISK_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def _load_state() -> RiskState:
    if not RISK_STATE_JSON.exists():
        return RiskState()
    try:
        data = json.load(open(RISK_STATE_JSON, "r", encoding="utf-8"))
        return RiskState(**data)
    except Exception:
        return RiskState()


def _pnl_to_returns(df_trades: pd.DataFrame) -> pd.Series:
    if df_trades.empty:
        return pd.Series(dtype=float)
    df = df_trades.copy()
    if "date" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "ret" in df.columns:
        daily = df.groupby("date")["ret"].mean().sort_index()
    elif "pnl" in df.columns:
        daily = (df.groupby("date")["pnl"].sum() / 100000.0).sort_index()
    else:
        return pd.Series(dtype=float)
    return daily.dropna()


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def _rolling_realized_vol(daily_ret: pd.Series, window: int = 20) -> float:
    if len(daily_ret) < 2:
        return 0.0
    return float(daily_ret.tail(window).std(ddof=1)) * np.sqrt(252.0)


def _losing_streak_days(daily_ret: pd.Series) -> int:
    """Consecutive losing days at end of series."""
    if daily_ret.empty:
        return 0
    streak = 0
    for v in reversed(daily_ret.dropna().tolist()):
        if v < 0:
            streak += 1
        else:
            break
    return streak


def _read_sentiment_macro() -> Tuple[float, float]:
    avg_sent, mac_stress = 0.0, 0.0
    df_s = _safe_read_csv(SENT_PATH)
    if not df_s.empty and "sentiment" in df_s.columns:
        try:
            avg_sent = float(pd.to_numeric(df_s["sentiment"], errors="coerce").dropna().mean())
        except Exception:
            pass
    df_m = _safe_read_csv(MACRO_PATH)
    if not df_m.empty and "value" in df_m.columns:
        try:
            mac_stress = float(pd.to_numeric(df_m["value"], errors="coerce").dropna().std())
        except Exception:
            pass
    return avg_sent, mac_stress


# ---------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------
class RiskController:
    """
    Phase 8.4+ : regime detection + cool-down + volatility targeting.
    """

    def __init__(
        self,
        vol_thresholds=(0.12, 0.20, 0.35),
        dd_thresholds=(-0.05, -0.10, -0.20),
        sentiment_bias=True,
        macro_bias=True,
        cooldown_streak=3,
        cooldown_days=1,
        severe_dd_cooldown=-0.12,
        vol_target: Optional[float] = 0.18,
        vol_intensity_bounds: Tuple[float, float] = (0.40, 1.50),
        vol_lookback_days: int = 20,
    ):
        self.vol_th = vol_thresholds
        self.dd_th = dd_thresholds
        self.sentiment_bias = sentiment_bias
        self.macro_bias = macro_bias
        self.cooldown_streak = cooldown_streak
        self.cooldown_days = cooldown_days
        self.severe_dd_cooldown = severe_dd_cooldown
        self.vol_target = vol_target
        self.vol_intensity_bounds = vol_intensity_bounds
        self.vol_lookback_days = vol_lookback_days
        self.state = _load_state()

    # -------------------------------------------------
    # internal helpers
    # -------------------------------------------------
    def _regime_from_stats(self, rv: float, dd60: float) -> str:
        if rv < self.vol_th[0] and dd60 > self.dd_th[0]:
            return "Calm"
        if rv < self.vol_th[1] and dd60 > self.dd_th[1]:
            return "Normal"
        if rv < self.vol_th[2] and dd60 > self.dd_th[2]:
            return "Volatile"
        return "Crisis"

    def _regime_bias(self, regime: str, sent: float, mac: float) -> str:
        r = regime
        if self.sentiment_bias and sent < -0.15:
            r = "Crisis" if r == "Volatile" else r
        if self.macro_bias:
            if mac >= 2.0 and r == "Normal":
                r = "Volatile"
            if mac >= 3.0:
                r = "Crisis"
        return r

    def _sizing_for_regime(self, regime: str) -> Tuple[float, float, float]:
        """Return stop_loss, take_profit, order_intensity base."""
        if regime == "Calm":
            return (0.03, 0.08, 1.10)
        if regime == "Normal":
            return (0.03, 0.06, 1.00)
        if regime == "Volatile":
            return (0.025, 0.05, 0.70)
        return (0.02, 0.04, 0.45)

    def _cooldown_needed(self, daily_ret: pd.Series, dd60: float) -> bool:
        if daily_ret.empty:
            return False
        if _losing_streak_days(daily_ret) >= self.cooldown_streak:
            return True
        if dd60 <= self.severe_dd_cooldown:
            return True
        return False

    # -------------------------------------------------
    # compute live state
    # -------------------------------------------------
    def compute_state(self) -> RiskState:
        trades = _safe_read_csv(TRADES_CSV)
        daily_ret = _pnl_to_returns(trades)
        rv = _rolling_realized_vol(daily_ret, window=self.vol_lookback_days)
        dd60 = _max_drawdown(daily_ret.tail(60))
        sent, mac = _read_sentiment_macro()

        regime0 = self._regime_from_stats(rv, dd60)
        regime = self._regime_bias(regime0, sent, mac)

        # base exposure multipliers
        if regime == "Calm":
            lev, mpos, cash = 1.20, 1.20, 0.00
        elif regime == "Normal":
            lev, mpos, cash = 1.00, 1.00, 0.02
        elif regime == "Volatile":
            lev, mpos, cash = 0.65, 0.70, 0.06
        else:
            lev, mpos, cash = 0.35, 0.40, 0.15

        circuit = (dd60 <= -0.25) or (rv >= 0.45)

        # base sizing
        sl, tp, intensity = self._sizing_for_regime(regime)

        # === Volatility targeting ===
        if self.vol_target and rv > 0:
            vt_factor = float(self.vol_target / rv)
            lo, hi = self.vol_intensity_bounds
            vt_factor = float(np.clip(vt_factor, lo, hi))
            intensity *= vt_factor
        else:
            vt_factor = 1.0

        # === Cool-down logic ===
        cooldown_active = False
        cooldown_until_iso = ""
        now = pd.Timestamp.now(tz="UTC")
        if self.state.cooldown_active and self.state.cooldown_until_iso:
            try:
                until = dt.datetime.fromisoformat(self.state.cooldown_until_iso)
                if now < until:
                    cooldown_active = True
                    cooldown_until_iso = self.state.cooldown_until_iso
            except Exception:
                pass

        if not cooldown_active and self._cooldown_needed(daily_ret, dd60):
            until = now + dt.timedelta(days=self.cooldown_days)
            cooldown_active = True
            cooldown_until_iso = until.isoformat()

        # persist state
        self.state = RiskState(
            regime=regime,
            realized_vol_20d=float(rv),
            max_drawdown_60d=float(dd60),
            avg_sentiment=float(sent),
            macro_stress=float(mac),
            leverage_mult=float(lev),
            maxpos_mult=float(mpos),
            cash_floor_add=float(cash),
            circuit_breaker=bool(circuit),
            stop_loss_pct=float(sl),
            take_profit_pct=float(tp),
            order_intensity=float(intensity),
            cooldown_active=bool(cooldown_active),
            cooldown_until_iso=cooldown_until_iso,
        )
        _save_state(self.state)
        return self.state

    # -------------------------------------------------
    # public accessors
    # -------------------------------------------------
    def get_multipliers(self) -> Tuple[float, float, float, bool]:
        """Return leverage_mult, maxpos_mult, cash_floor_add, circuit_breaker"""
        return (
            self.state.leverage_mult,
            self.state.maxpos_mult,
            self.state.cash_floor_add,
            self.state.circuit_breaker,
        )

    def get_order_sizing(self) -> Tuple[float, float, float, bool, str]:
        """Return stop_loss, take_profit, order_intensity, cooldown_active, cooldown_until"""
        return (
            self.state.stop_loss_pct,
            self.state.take_profit_pct,
            self.state.order_intensity,
            self.state.cooldown_active,
            self.state.cooldown_until_iso,
        )
