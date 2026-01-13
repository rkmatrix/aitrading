from __future__ import annotations
from typing import Dict
import pandas as pd
from .cross_asset_factors import build_cross_asset_features

REGISTRY = {
    "xasset_momentum": build_cross_asset_features,
    "xasset_meanrev": build_cross_asset_features,
    "macro_regime": build_cross_asset_features,
}

def build_all(names, ohlcv: Dict[str, pd.DataFrame], params_list: list[dict]) -> Dict[str, pd.DataFrame]:
    out = {sym: pd.DataFrame(index=df.index) for sym, df in ohlcv.items()}
    for name, params in zip(names, params_list):
        fn = REGISTRY.get(name)
        if not fn:
            continue
        feat = fn(ohlcv, name=name, **(params or {}))
        for sym, df in feat.items():
            out[sym] = out[sym].join(df, how="outer")
    return out
