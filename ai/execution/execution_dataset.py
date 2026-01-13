from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List

from .orderlog_parser import parse_logs

class ExecDataset:
    def __init__(self, paths: List[str]):
        self.paths = paths
        self.df: pd.DataFrame | None = None

    def build(self) -> pd.DataFrame:
        df = parse_logs(self.paths)
        # Keep essential columns
        keep = [
            "timestamp", "symbol", "side", "qty",
            "order_type", "limit_price", "fill_price",
            "status", "fees", "slippage_bps", "latency_ms",
            "best_bid", "best_ask", "mid", "spread",
            "order_id", "parent_id"
        ]
        self.df = df[[c for c in keep if c in df.columns]].copy()
        return self.df

    def cache(self, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        assert self.df is not None, "Call build() before cache()"
        self.df.to_parquet(out_path, index=False)
