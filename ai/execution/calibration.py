from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

class ExecCalibrator:
    def __init__(self, spread_floor_bps: float = 0.2):
        self.spread_floor_bps = spread_floor_bps
        self.params: Dict[str, Any] = {}

    def fit(self, logs: pd.DataFrame) -> Dict[str, Any]:
        df = logs.copy()
        df = df[df["status"].isin(["filled", "partial", "canceled", "rejected", "open", "new"])].copy()
        df = df[df["qty"].abs() > 0]

        # spread in bps relative to mid
        if {"spread", "mid"}.issubset(df.columns):
            df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
            df["spread_bps"] = df["spread_bps"].clip(lower=self.spread_floor_bps)
        else:
            df["spread_bps"] = self.spread_floor_bps

        # slippage stats from fills
        filled = df[(df["status"] == "filled") & df["fill_price"].notna() & df["mid"].notna()]
        if not filled.empty:
            side_sign = np.where(filled["side"].eq("buy"), 1, -1)
            impl_shortfall = side_sign * ((filled["fill_price"] - filled["mid"]) / filled["mid"]) * 1e4
            a_bps = float(np.nanmedian(np.abs(impl_shortfall)))
            sigma_bps = float(np.nanstd(impl_shortfall))
        else:
            a_bps, sigma_bps = 3.0, 2.0

        lat_mean = float(np.nanmedian(df.get("latency_ms", pd.Series([120]))))
        lat_std = float(np.nanstd(df.get("latency_ms", pd.Series([80]))))

        if not filled.empty:
            med_spread = float(np.nanmedian(df["spread_bps"])) if "spread_bps" in df else 1.0
            base_passive = max(0.1, min(0.6, 0.4 * (1.0 / (med_spread / max(1e-6, self.spread_floor_bps)))))
        else:
            base_passive = 0.35

        self.params = {
            "impact": {"a_bps": a_bps, "sigma_bps": sigma_bps},
            "latency": {"mean_ms": lat_mean, "std_ms": lat_std},
            "passive_fill_prob": base_passive,
            "spread_floor_bps": self.spread_floor_bps
        }
        return self.params

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)
