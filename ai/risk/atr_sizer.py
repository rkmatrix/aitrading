# ai/risk/atr_sizer.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from infra.marketdata.ohlcv import get_ohlcv

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = np.maximum.reduce([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ])
    return tr.rolling(period, min_periods=period).mean()

def atr_sizing_for_symbol(
    symbol: str,
    interval: str = "1d",
    atr_period: int = 14,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> Tuple[float, float, Optional[float]]:
    """
    Returns (stop_loss_pct, take_profit_pct, last_close)
    pct values are relative to last close, always positive (e.g., 0.025 = 2.5%)
    """
    end = pd.Timestamp.utcnow()
    df = get_ohlcv(symbol, end=end, interval=interval)
    if df.empty or len(df) < atr_period + 2:
        return (0.03, 0.06, None)  # safe default
    a = atr(df, atr_period).iloc[-1]
    px = float(df["Close"].iloc[-1])
    if px <= 0 or pd.isna(a) or a <= 0:
        return (0.03, 0.06, px)
    sl_pct = float(sl_mult * a / px)
    tp_pct = float(tp_mult * a / px)
    # clamp to sane ranges
    sl_pct = float(np.clip(sl_pct, 0.005, 0.05))  # 0.5% .. 5%
    tp_pct = float(np.clip(tp_pct, 0.01, 0.12))   # 1% .. 12%
    return (sl_pct, tp_pct, px)

def batch_atr_sizing(symbols, **kwargs) -> Dict[str, Tuple[float, float, Optional[float]]]:
    out = {}
    for s in symbols:
        out[s] = atr_sizing_for_symbol(s, **kwargs)
    return out
