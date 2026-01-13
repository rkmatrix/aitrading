from __future__ import annotations
from typing import Dict
import pandas as pd

class FeatureBuilder:
    """Base contract for feature builders."""
    def build(self, ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError
