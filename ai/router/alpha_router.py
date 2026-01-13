import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from ai.allocators.meta_ppo.constraints import project_simplex, clamp_trust_region


@dataclass
class RouterConfig:
    per_asset_cap: float = 0.25
    gross_cap: float = 1.25
    net_cap: float = 0.75
    trust_region_l1: float = 0.15


class AlphaRouter:
    """Blend alpha signals with allocator weights."""

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.prev = None

    def _cap(self, w: Dict[str, float]) -> Dict[str, float]:
        w = {k: np.clip(v, -self.cfg.per_asset_cap, self.cfg.per_asset_cap) for k, v in w.items()}
        gross = sum(abs(v) for v in w.values())
        net = abs(sum(w.values()))
        scale = min(1.0, self.cfg.gross_cap / (gross + 1e-12), self.cfg.net_cap / (net + 1e-12))
        return {k: v * scale for k, v in w.items()}

    def blend(self, signals: Dict[str, float], weights: np.ndarray, assets: List[str]) -> Dict[str, float]:
        weights = project_simplex(weights)
        s_vec = np.array([signals.get(a, 0.0) for a in assets])
        s_norm = s_vec / (np.sum(np.abs(s_vec)) + 1e-12)
        w_signed = weights * np.sign(s_norm) * np.minimum(1.0, np.abs(s_norm))
        out = {a: float(w_signed[i]) for i, a in enumerate(assets)}
        prev = np.array([self.prev.get(a, 0.0) if self.prev else 0.0 for a in assets])
        cur  = np.array(list(out.values()))
        
        # Always enforce trust region (even first step)
        cur = clamp_trust_region(cur, prev, self.cfg.trust_region_l1)
        out = {a: float(cur[i]) for i, a in enumerate(assets)}
        
        self.prev = out
        return self._cap(out)

