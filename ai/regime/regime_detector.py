# ai/regime/regime_detector.py (UPDATED FOR PHASE 84)
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MarketRegimeDetector")


@dataclass
class RegimeDetectionConfig:
    price_history_file: Path
    lookback_bars: int
    min_bars: int
    date_column: str
    close_column: str
    vol_low: float
    vol_high: float
    trend_up: float
    trend_down: float


@dataclass
class RegimeResult:
    regime: str
    vol: float
    slope: float
    details: Dict[str, Any]


class MarketRegimeDetector:
    """
    Daily regime classifier (unchanged).

    Classifies:
        - risk_on
        - risk_off
        - neutral
    """

    def __init__(self, cfg: Dict[str, Any]):
        rd = cfg.get("regime_detection", {})
        paths = cfg.get("paths", {})

        self.cfg = RegimeDetectionConfig(
            price_history_file=Path(paths.get("price_history_file", "data/market/spy_daily.csv")),
            lookback_bars=int(rd.get("lookback_bars", 60)),
            min_bars=int(rd.get("min_bars", 30)),
            date_column=rd.get("date_column", "date"),
            close_column=rd.get("close_column", "close"),
            vol_low=float(rd.get("vol_thresholds", {}).get("low", 0.01)),
            vol_high=float(rd.get("vol_thresholds", {}).get("high", 0.03)),
            trend_up=float(rd.get("trend_thresholds", {}).get("up", 0.001)),
            trend_down=float(rd.get("trend_thresholds", {}).get("down", -0.001)),
        )

    # -----------------------------
    # Data loading
    # -----------------------------
    def _load_prices(self) -> pd.DataFrame:
        if not self.cfg.price_history_file.exists():
            raise FileNotFoundError(f"Price history file not found: {self.cfg.price_history_file}")

        df = pd.read_csv(self.cfg.price_history_file)
        if self.cfg.date_column not in df.columns:
            raise KeyError(f"Date column '{self.cfg.date_column}' not in CSV.")
        if self.cfg.close_column not in df.columns:
            raise KeyError(f"Close column '{self.cfg.close_column}' not in CSV.")

        df = df[[self.cfg.date_column, self.cfg.close_column]].copy()
        df = df.dropna()
        df = df.sort_values(self.cfg.date_column)
        df = df.reset_index(drop=True)
        return df

    # -----------------------------
    # Core regime logic
    # -----------------------------
    def detect_regime(self) -> RegimeResult:
        df = self._load_prices()
        if len(df) < self.cfg.min_bars:
            raise ValueError(
                f"Not enough history for regime detection: have {len(df)}, "
                f"need at least {self.cfg.min_bars}."
            )

        df_tail = df.tail(self.cfg.lookback_bars).copy()
        closes = df_tail[self.cfg.close_column].astype(float).to_numpy()

        # Daily returns
        rets = np.diff(np.log(closes))
        vol = float(np.std(rets)) if len(rets) > 0 else 0.0

        # Trend slope (simple linear fit)
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0]) if len(closes) >= 2 else 0.0

        # Trend labeling
        if slope >= self.cfg.trend_up:
            trend_label = "up"
        elif slope <= self.cfg.trend_down:
            trend_label = "down"
        else:
            trend_label = "flat"

        # Volatility labeling
        if vol <= self.cfg.vol_low:
            vol_label = "low"
        elif vol >= self.cfg.vol_high:
            vol_label = "high"
        else:
            vol_label = "mid"

        # Combine
        if trend_label == "up" and vol_label in ("low", "mid"):
            regime = "risk_on"
        elif trend_label == "down" and vol_label in ("mid", "high"):
            regime = "risk_off"
        else:
            regime = "neutral"

        details = {
            "trend_label": trend_label,
            "vol_label": vol_label,
            "last_date": df_tail[self.cfg.date_column].iloc[-1],
            "bars_used": len(df_tail),
        }

        logger.info(
            "Regime detection → regime=%s | vol=%.6f (%s) | slope=%.6f (%s)",
            regime, vol, vol_label, slope, trend_label,
        )

        return RegimeResult(
            regime=regime,
            vol=vol,
            slope=slope,
            details=details,
        )


# ============================================================
# 🚀 PHASE 84 — INTRADAY REGIME DETECTOR (LIVE)
# ============================================================

def detect_intraday_regime(prices: np.ndarray) -> str:
    """
    Lightweight, fast regime classifier for intraday 1m bars.

    Expected row layout (matching ExecutionAwareLiveAgent):
        [open, high, low, close, volume]
    Input: prices shape (window, 5)
    Outputs:
        - "quiet_trend"
        - "volatile_trend"
        - "rangebound"
        - "chaos"
        - "extreme_vol"
    """
    if prices is None or len(prices) < 20:
        return "unknown"

    # Our layout: [open, high, low, close, volume]
    open_ = prices[:, 0]
    high = prices[:, 1]
    low = prices[:, 2]
    close = prices[:, 3]

    # ATR% approximation
    if close[-1] == 0:
        return "unknown"
    atr_pct = float((high.max() - low.min()) / abs(close[-1]))

    # Trend strength via EMA spread on closes
    ema10 = close[-10:].mean()
    ema30 = close[-30:].mean() if len(close) >= 30 else close.mean()
    trend = float(ema10 - ema30)

    # Classification logic
    if atr_pct < 0.005 and abs(trend) > 0.002:
        return "quiet_trend"

    if atr_pct > 0.01 and abs(trend) > 0.003:
        return "volatile_trend"

    if atr_pct > 0.03:
        return "extreme_vol"

    if abs(trend) < 0.001 and atr_pct < 0.015:
        return "rangebound"

    return "chaos"

