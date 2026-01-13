from __future__ import annotations
import pandas as pd

def make_time_index_tzaware(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Ensure tz-awareness and convert to provided tz.
    Avoids 'tz-naive vs tz-aware' crashes that were popping up earlier.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    # Convert to desired tz but keep arithmetic safe
    return df.tz_convert(tz)

def resample_to_timebars(df: pd.DataFrame, rule: str = "1S") -> pd.DataFrame:
    """
    Simple resample for L1 to fixed time bars (e.g., '1S', '100ms').
    """
    close_mid = ((df["bid"] + df["ask"]) / 2.0).rename("mid")
    out = pd.DataFrame({
        "bid": df["bid"].resample(rule).last(),
        "ask": df["ask"].resample(rule).last(),
        "bid_size": df.get("bid_size", 0).resample(rule).last(),
        "ask_size": df.get("ask_size", 0).resample(rule).last(),
        "mid": close_mid.resample(rule).last()
    }).dropna(subset=["bid", "ask"])
    out["volatility"] = out["mid"].pct_change().rolling(100, min_periods=10).std().fillna(method="bfill").fillna(0.002)
    out["arrival_price"] = out["mid"]
    out["est_volume"] = 1000.0
    return out
