# ai/policy/evaluator.py
from __future__ import annotations
import numpy as np, pandas as pd, random
from typing import Dict, Any

class QuickEvaluator:
    """
    Placeholder fast evaluation. In your integration, replace `score_policy`
    with a call into your backtest/sim tick harness that yields a metric.
    """
    def __init__(self, window_bars: int = 200, seed: int = 43, reward_metric: str = "sharpe", fail_score: float = -1e9, allow_metadata_only: bool = True):
        self.window_bars = window_bars
        self.seed = seed
        self.reward_metric = reward_metric
        self.fail_score = fail_score
        self.allow_metadata_only = allow_metadata_only

    def score_policy(self, bundle_manifest: Dict[str, Any], has_weights: bool) -> float:
        # Example: if no weights and not allowed → fail_score
        if not has_weights and not self.allow_metadata_only:
            return self.fail_score

        # Dummy stochastic score derived from hyperparams for determinism
        rnd_seed = hash(str(bundle_manifest.get("hyperparams", {}))) ^ self.seed
        rnd = random.Random(rnd_seed)
        base = rnd.uniform(-0.2, 0.2)
        # Heuristic: prefer reasonable lr/gamma/entropy ranges
        hp = bundle_manifest.get("hyperparams", {})
        lr = float(hp.get("lr", 3e-4))
        gamma = float(hp.get("gamma", 0.99))
        entropy = float(hp.get("entropy_coef", 0.0))
        clip = float(hp.get("clip_range", 0.2))

        bonus = 0.0
        if 1e-5 <= lr <= 5e-3: bonus += 0.10
        if 0.9 <= gamma <= 0.999: bonus += 0.05
        if 0.0 <= entropy <= 0.05: bonus += 0.05
        if 0.05 <= clip <= 0.4: bonus += 0.05

        score = base + bonus
        # Map to chosen metric space (e.g., Sharpe proxy)
        if self.reward_metric == "sharpe":
            return float(score)
        elif self.reward_metric == "winrate":
            return float(0.5 + 0.5*max(min(score, 0.5), -0.5))
        elif self.reward_metric == "pnl":
            return float(1000.0*score)
        return float(score)
