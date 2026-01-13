# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np

@dataclass
class FeatureConfig:
    lookback: int = 10

class FeatureFeeder:
    """Builds state vectors from recent microprice/volume and inventory/time context."""
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build_state(self, mid_hist, vol_hist, inventory_frac, time_frac):
        mid = np.array(mid_hist[-self.cfg.lookback:], dtype=np.float32)
        vol = np.array(vol_hist[-self.cfg.lookback:], dtype=np.float32)
        if mid.size < self.cfg.lookback:
            mid = np.pad(mid, (self.cfg.lookback - mid.size, 0))
        if vol.size < self.cfg.lookback:
            vol = np.pad(vol, (self.cfg.lookback - vol.size, 0))
        mid = (mid - np.nanmean(mid)) / (np.nanstd(mid) + 1e-6)
        vol = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-6)
        ctx = np.array([inventory_frac, time_frac], dtype=np.float32)
        return np.concatenate([mid, vol, ctx], axis=0)
