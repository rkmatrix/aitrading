# ai/eval/shadow_evaluator.py
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO

log = logging.getLogger(__name__)


@dataclass
class ShadowEvalRow:
    timestamp: float
    symbol: str
    live_action: float
    shadow_action: float
    price: float
    pnl_live: float
    pnl_shadow: float
    divergence: float


class ShadowEvaluator:
    """
    Compares live vs shadow policy actions & PnL without sending shadow orders.

    You call:
      evaluator.record_step(obs, price, live_action, shadow_action, pnl_live, pnl_shadow)
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "symbol",
                        "live_action",
                        "shadow_action",
                        "price",
                        "pnl_live",
                        "pnl_shadow",
                        "divergence",
                    ],
                )
                writer.writeheader()

    @staticmethod
    def load_policy(path: str, device: str = "cpu") -> PPO:
        log.info("Loading PPO policy from %s", path)
        return PPO.load(path, device=device, print_system_info=False)

    def record_step(self, row: ShadowEvalRow) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "symbol",
                    "live_action",
                    "shadow_action",
                    "price",
                    "pnl_live",
                    "pnl_shadow",
                    "divergence",
                ],
            )
            writer.writerow(asdict(row))
