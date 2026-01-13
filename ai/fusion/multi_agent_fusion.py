# ai/fusion/multi_agent_fusion.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Iterable

log = logging.getLogger(__name__)


@dataclass
class FusedRow:
    timestamp: float
    symbol: str
    alpha_action: float
    sentiment_score: float
    volatility_score: float
    fused_signal: float


class MultiAgentFusionEngine:
    """
    Fuses multiple agent outputs into a single continuous target signal.
    """

    def __init__(self, alpha_weight: float, sentiment_weight: float, vol_weight: float) -> None:
        self.alpha_w = alpha_weight
        self.sent_w = sentiment_weight
        self.vol_w = vol_weight

    def fuse(self, alpha_action: float, sentiment: float, vol_score: float) -> float:
        # vol_score is often "risk": higher risk => smaller position; invert it
        safe_vol = 1.0 / (1.0 + max(vol_score, 0.0))
        return (
            self.alpha_w * alpha_action +
            self.sent_w * sentiment +
            self.vol_w * safe_vol
        )


def load_alpha_replay(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        log.warning("Alpha replay not found: %s", p)
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_csv_map(path: str, key_field: str = "timestamp") -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        log.warning("CSV not found: %s", p)
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = str(row[key_field])
            out[k] = row
    return out


def write_fused_csv(path: str, rows: Iterable[FusedRow]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "symbol",
                "alpha_action",
                "sentiment_score",
                "volatility_score",
                "fused_signal",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
