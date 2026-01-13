from __future__ import annotations
from typing import Dict, Iterable
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from .provider_base import MarketDataProvider

class YahooProvider(MarketDataProvider):
    def get_ohlcv(self, symbols: Iterable[str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if yf is None:
            for s in symbols:
                out[s] = pd.DataFrame(columns=["Open","High","Low","Close","Adj Close","Volume"]).astype(float)
            return out
        for s in symbols:
            try:
                df = yf.download(s, period=period, interval=interval, auto_adjust=True, progress=False)
                if not df.empty:
                    df = df.rename(columns={"Adj Close": "AdjClose"})
                out[s] = df
            except Exception:
                out[s] = pd.DataFrame()
        return out
