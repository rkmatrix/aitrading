# ai/stability/meta_stability_engine.py
# Phase 123 – Meta-Stability Engine
#
# This engine monitors:
#   • micro-term stability  (3 ticks)
#   • mid-term drift        (30 ticks)
#   • volatility cycles     (150 ticks)
#   • regime hysteresis     (freeze conditions)
#   • dynamic clamp factor  (0.4–1.0)
#
# API:
#   update(price: float)
#   compute() -> dict:
#       {
#           "stability_score": float in [0,1],
#           "clamp_factor": float in [0.4,1.0],
#           "decision_freeze": bool
#       }


from __future__ import annotations
import numpy as np
from collections import deque
import logging

log = logging.getLogger(__name__)


class MetaStabilityEngine:
    """
    Tracks market stability across 3, 30, and 150 tick windows.
    Produces:
        - stability_score ∈ [0,1]
        - clamp_factor    ∈ [0.4,1.0]
        - decision_freeze ∈ {True, False}
    """

    def __init__(self, mode: str = "moderate") -> None:
        """
        mode:
          - moderate (recommended)
          - strong
          - ultra
        """
        self.mode = mode

        # Tick windows
        self.win_small = deque(maxlen=3)
        self.win_mid = deque(maxlen=30)
        self.win_big = deque(maxlen=150)

        # Freeze thresholds adjust by mode
        if mode == "ultra":
            self.freeze_z_thresh = 2.5
            self.clamp_floor = 0.30
        elif mode == "strong":
            self.freeze_z_thresh = 3.0
            self.clamp_floor = 0.40
        else:  # moderate
            self.freeze_z_thresh = 3.5
            self.clamp_floor = 0.40

    # --------------------------------------------------------------
    # Update with latest price
    # --------------------------------------------------------------
    def update(self, price: float) -> None:
        try:
            p = float(price)
        except Exception:
            return

        self.win_small.append(p)
        self.win_mid.append(p)
        self.win_big.append(p)

    # --------------------------------------------------------------
    # Compute stability metrics
    # --------------------------------------------------------------
    def _zscore(self, arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        mu = arr.mean()
        sigma = arr.std() + 1e-9
        return float(abs(arr[-1] - mu) / sigma)

    def _compute_window_stability(self, window: deque) -> float:
        """Return stability ∈ [0,1] based on volatility."""
        if len(window) < 3:
            return 1.0
        arr = np.array(window, dtype=float)
        vol = arr.std()
        if vol <= 0:
            return 1.0
        # higher vol -> lower stability (log scaled)
        score = 1.0 / (1.0 + np.log1p(vol))
        return float(max(0.0, min(1.0, score)))

    # --------------------------------------------------------------
    # Public compute() interface called by Phase 26
    # --------------------------------------------------------------
    def compute(self) -> dict:
        """
        Returns:
            stability_score ∈ [0,1]
            clamp_factor    ∈ [clamp_floor, 1.0]
            decision_freeze ∈ bool
        """

        # Compute z-score hysteresis
        big_arr = np.array(self.win_big, dtype=float)
        mid_arr = np.array(self.win_mid, dtype=float)

        z_big = self._zscore(big_arr) if len(big_arr) > 5 else 0.0
        z_mid = self._zscore(mid_arr) if len(mid_arr) > 5 else 0.0

        # Volatility-based stability
        s_small = self._compute_window_stability(self.win_small)
        s_mid = self._compute_window_stability(self.win_mid)
        s_big = self._compute_window_stability(self.win_big)

        # Weighted combination
        stability = float(0.5 * s_small + 0.3 * s_mid + 0.2 * s_big)

        # Clamp factor (inverse of instability)
        clamp = max(self.clamp_floor, min(1.0, stability))

        # Freeze logic (only during extreme z-score shocks)
        freeze = (z_big > self.freeze_z_thresh) or (z_mid > self.freeze_z_thresh)

        result = {
            "stability_score": float(stability),
            "clamp_factor": float(clamp),
            "decision_freeze": bool(freeze),
        }

        return result
