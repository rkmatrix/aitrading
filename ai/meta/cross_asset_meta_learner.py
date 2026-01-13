from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd

from ai.utils.config import AppConfig
from ai.router.alpha_router import AlphaRouter
from ai.meta.signal_aggregator import SignalAggregator
from ai.meta.risk_coordinator import RiskCoordinator

@dataclass
class MetaDecision:
    weights: Dict[str, float]
    diagnostics: Dict[str, float]

class CrossAssetMetaLearner:
    """Top-level orchestrator for cross-asset allocation.

    Steps per rebalance:
      1) Pull/prepare features per asset.
      2) Route to experts → raw signals.
      3) Aggregate signals → unified scores.
      4) Apply risk/constraints → final weights.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.router = AlphaRouter(cfg)
        self.aggregator = SignalAggregator(cfg)
        self.risk = RiskCoordinator(cfg)

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        if hasattr(self.router, "fit"):
            self.router.fit(data)
        if hasattr(self.aggregator, "fit"):
            self.aggregator.fit(data)
        if hasattr(self.risk, "fit"):
            self.risk.fit(data)

    def decide(self, features: Dict[str, pd.DataFrame]) -> MetaDecision:
        raw = self.router.signals(features)
        agg = self.aggregator.combine(raw)
        weights = self.risk.target(agg)
        return MetaDecision(weights=weights, diagnostics={"n_assets": len(weights)})
