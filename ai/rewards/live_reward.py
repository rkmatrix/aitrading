# ai/rewards/live_reward.py
from __future__ import annotations
from typing import Optional, Dict, Any

def step_reward_from_fill(row: Dict[str, Any],
                          risk_penalty: float = 0.0,
                          commission: float = 0.0) -> float:
    """
    Compute a per-step reward from a fill row (or executor event dict).
    Expects keys: 'pnl' (preferred) or derive from qty*price deltas externally.
    Optional: apply a tiny risk/commission penalty if desired.
    """
    pnl = float(row.get("pnl", 0.0))
    reward = pnl
    if commission:
        reward -= float(commission)
    if risk_penalty:
        vol = float(row.get("volatility", 0.0))
        reward -= risk_penalty * vol
    return float(reward)
