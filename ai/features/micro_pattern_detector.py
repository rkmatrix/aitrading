# ai/features/micro_pattern_detector.py

from __future__ import annotations

import numpy as np
from typing import Dict, Any


def compute_micro_features(window: np.ndarray) -> Dict[str, float]:
    """
    Compute micro-pattern features from a rolling OHLCV window.

    Expected shape: (N, 5) with columns:
        [open, high, low, close, volume]

    Returns a dict of scalar features such as:
        - short/medium momentum
        - volatility ratios
        - volume spike ratio
        - intraday range
        - price position in recent range
    """
    feats: Dict[str, float] = {}

    if window is None or window.ndim != 2 or window.shape[1] < 4:
        return feats

    # Our layout: [open, high, low, close, volume]
    close = window[:, 3]
    high = window[:, 1]
    low = window[:, 2]
    vol = window[:, 4]

    n = window.shape[0]
    if n < 3:
        return feats

    last_close = float(close[-1])

    # --- Momentum over different horizons ---
    def safe_mom(offset: int) -> float:
        if n <= offset or close[-offset - 1] == 0:
            return 0.0
        return float(last_close / close[-offset - 1] - 1.0)

    feats["mom_1"] = safe_mom(1)   # last bar vs previous
    feats["mom_5"] = safe_mom(5)   # last vs 5 bars ago
    feats["mom_10"] = safe_mom(10) # last vs 10 bars ago

    # --- Volatility metrics (log returns) ---
    def realized_vol(series: np.ndarray) -> float:
        if series.size < 2:
            return 0.0
        rets = np.diff(np.log(series + 1e-8))
        return float(np.std(rets)) if rets.size > 0 else 0.0

    vol_20 = realized_vol(close[-20:]) if n >= 20 else realized_vol(close)
    vol_60 = realized_vol(close)

    feats["vol_20"] = vol_20
    feats["vol_60"] = vol_60
    feats["vol_ratio_20_60"] = float(vol_20 / (vol_60 + 1e-8) if vol_60 > 0 else 1.0)

    # --- Volume spike ---
    if n >= 20:
        avg_vol_20 = float(np.mean(vol[-20:]) + 1e-3)
    else:
        avg_vol_20 = float(np.mean(vol) + 1e-3)
    feats["volume_ratio"] = float(vol[-1] / avg_vol_20)

    # --- Intraday range on last bar ---
    last_high = float(high[-1])
    last_low = float(low[-1])
    if last_close != 0:
        feats["intraday_range_norm"] = float((last_high - last_low) / abs(last_close))
    else:
        feats["intraday_range_norm"] = 0.0

    # --- Price position in recent range ---
    min_low = float(low.min())
    max_high = float(high.max())
    denom = max_high - min_low
    if denom <= 0:
        feats["price_pos_in_range"] = 0.5
    else:
        feats["price_pos_in_range"] = float((last_close - min_low) / denom)

    # --- Simple breakout / squeeze style proxies ---
    # Breakout-ish if price is near top of range and momentum positive
    feats["breakout_score"] = float(
        max(0.0, feats["price_pos_in_range"] - 0.7) * max(0.0, feats["mom_5"])
    )

    # Volatility compression if recent range and vol are small
    feats["squeeze_score"] = float(
        1.0 / (1.0 + feats["intraday_range_norm"] * (1.0 + vol_20))
    )

    return feats
