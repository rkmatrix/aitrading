# ai/features/feature_factory.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    returns_windows: List[int]
    vol_windows: List[int]
    ma_windows: List[int]
    rsi_windows: List[int]
    symbol_col: str = "symbol"
    price_col: str = "close"
    volume_col: str = "volume"


class FeatureFactory:
    """
    Phase 101 – Feature Factory

    Given a DataFrame with columns:
        ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']

    It produces per-row features like:
        - normalized returns over multiple windows
        - rolling volatility
        - moving averages
        - RSI
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    def transform_symbol_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: DataFrame for a single symbol, sorted by timestamp ascending.
        """
        cfg = self.cfg
        price = df[cfg.price_col].astype(float)

        out = df.copy()

        # Returns
        for w in cfg.returns_windows:
            out[f"ret_{w}"] = price.pct_change(periods=w, fill_method=None)

        # Volatility (std dev of returns)
        for w in cfg.vol_windows:
            out[f"vol_{w}"] = out[f"ret_{1}"].rolling(window=w).std() if "ret_1" in out.columns else price.pct_change().rolling(window=w).std()

        # Moving averages
        for w in cfg.ma_windows:
            out[f"ma_{w}"] = price.rolling(window=w).mean()
            out[f"ma_ratio_{w}"] = price / (out[f"ma_{w}"] + 1e-9)

        # RSI
        for w in cfg.rsi_windows:
            out[f"rsi_{w}"] = self._rsi(price, window=w)

        # Forward return (target)
        out["fwd_ret_1"] = price.shift(-1) / price - 1.0

        # Drop initial NaNs
        out = out.dropna().reset_index(drop=True)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature generation per symbol and concat.

        df must have a 'symbol' column.
        """
        cfg = self.cfg
        if cfg.symbol_col not in df.columns:
            raise ValueError(f"Expected symbol column '{cfg.symbol_col}' in input df")

        frames = []
        for sym, g in df.groupby(cfg.symbol_col):
            g_sorted = g.sort_values("timestamp")
            out_sym = self.transform_symbol_df(g_sorted)
            frames.append(out_sym)
            log.info("FeatureFactory: generated features for %s (%d rows)", sym, len(out_sym))

        if not frames:
            return pd.DataFrame()

        out_df = pd.concat(frames, ignore_index=True)
        return out_df
