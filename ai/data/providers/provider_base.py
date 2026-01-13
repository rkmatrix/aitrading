from __future__ import annotations
from typing import Dict, Iterable
import pandas as pd

class MarketDataProvider:
    def get_ohlcv(self, symbols: Iterable[str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        raise NotImplementedError
