# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ExposureCaps:
    gross_exposure_cap: float = 1.5   # 150%
    net_exposure_cap: float = 0.8     # 80% net long
    per_symbol_cap: float = 0.6       # 60% per symbol
    sector_caps: Dict[str, float] | None = None  # e.g., {"Tech": 0.7}


@dataclass
class HardRisk:
    max_daily_var_pct: float = 4.0
    max_position_value_pct: float = 25.0
    halt_on_violation: bool = True


def clamp_weights(
    weights: Dict[str, float],
    caps: ExposureCaps,
    symbol_to_sector: Dict[str, str] | None = None
) -> Dict[str, float]:
    """Project weights into the feasible region given soft caps."""
    w = weights.copy()
    # Per-symbol cap
    for s in list(w.keys()):
        w[s] = float(np.clip(w[s], -caps.per_symbol_cap, caps.per_symbol_cap))

    # Sector caps (if provided)
    if caps.sector_caps and symbol_to_sector:
        sector_sum: Dict[str, float] = {}
        for sym, wt in w.items():
            sec = symbol_to_sector.get(sym, "UNKNOWN")
            sector_sum[sec] = sector_sum.get(sec, 0.0) + abs(wt)
        for sym, wt in list(w.items()):
            sec = symbol_to_sector.get(sym, "UNKNOWN")
            cap = caps.sector_caps.get(sec, 1.0)
            total = sector_sum[sec]
            if total > cap and total > 0:
                # scale down sector proportionally
                scale = cap / total
                w[sym] = wt * scale

    # Gross exposure cap
    gross = sum(abs(v) for v in w.values())
    if gross > caps.gross_exposure_cap and gross > 0:
        scale = caps.gross_exposure_cap / gross
        for s in w:
            w[s] *= scale

    # Net exposure cap
    net = sum(w.values())
    if abs(net) > caps.net_exposure_cap:
        adjust = caps.net_exposure_cap / net
        for s in w:
            w[s] *= adjust

    return w
