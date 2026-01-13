import pandas as pd
import numpy as np


def _ensure_tz_aware(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Normalize all to UTC tz-aware to avoid "Cannot subtract tz-naive and tz-aware"
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df.index = _ensure_tz_aware(df.index)
    df = df.sort_index()

    # ---- Example minimal features (extend with your Phase 3–6 set) ----
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_diff"] = (df["sma_10"] - df["sma_50"]) / (df["sma_50"].abs() + 1e-9)
    df = df.dropna().replace([np.inf, -np.inf], 0.0)
    return df


def build_live_row(latest_ohlcv: dict, last_window: pd.DataFrame) -> pd.DataFrame:
    """
    latest_ohlcv: {"ts": pd.Timestamp|str, "open": float, "high": float, "low": float, "close": float, "volume": float}
    last_window: last N rows used for rolling features
    Returns a 1-row DataFrame aligned + tz-aware with engineered features.
    """
    ts = pd.to_datetime(latest_ohlcv["ts"], utc=True)
    row = pd.DataFrame(
        {
            "open": [latest_ohlcv["open"]],
            "high": [latest_ohlcv["high"]],
            "low": [latest_ohlcv["low"]],
            "close": [latest_ohlcv["close"]],
            "volume": [latest_ohlcv["volume"]],
        },
        index=[ts],
    )
    tmp = pd.concat([last_window.tail(200)[["open", "high", "low", "close", "volume"]], row])
    feats = build_features(tmp).tail(1)
    return feats
