from __future__ import annotations
from typing import Dict
import pandas as pd

# Skeleton: simple returns to keep pipeline wired.
def build_cross_asset_features(ohlcv: Dict[str, pd.DataFrame], name: str, **kwargs) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv.items():
        if df.empty:
            out[sym] = pd.DataFrame(index=df.index)
            continue
        close = df["Close"].astype(float)
        feat = pd.DataFrame(index=df.index)
        feat[f"{name}_ret1"] = close.pct_change().fillna(0.0)
        feat[f"{name}_ret5"] = close.pct_change(5).fillna(0.0)
        out[sym] = feat
    return out
