"""
ai/reward/exec_reward_tuner.py

Phase 83 — Execution-Aware Reward Tuner

Takes aggregated Execution Alpha metrics (from ExecutionAlphaEngine.to_dict())
and suggests a set of reward weights for ExecutionAwareReward:

  - slippage_weight
  - spread_weight
  - latency_weight
  - fill_prob_weight
  - clamp_penalty
  - pred_block_penalty
  - risk_block_penalty

This lets the bot "self-learn" which aspects of execution to punish or reward
based on its own realized trading history.
"""

from __future__ import annotations
from typing import Dict, Any


def _aggregate_global_stats(alpha_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate global averages across all symbols/sides/brokers.
    """
    total_trades = 0
    total_slip_bps = 0.0
    total_slip_abs = 0.0
    total_lat = 0.0
    total_filled = 0
    total_count = 0

    for _sym, side_map in alpha_dict.items():
        for _side, broker_map in side_map.items():
            for _broker, stats in broker_map.items():
                c = int(stats.get("count", 0))
                fc = int(stats.get("filled_count", 0))
                if c <= 0:
                    continue

                total_trades += c
                total_count += c
                total_filled += fc
                total_slip_bps += float(stats.get("avg_slippage_bps", 0.0)) * c
                total_slip_abs += float(stats.get("avg_slippage_abs", 0.0)) * c
                total_lat += float(stats.get("avg_latency_sec", 0.0)) * c

    if total_count == 0:
        # No data yet -> neutral stats
        return {
            "avg_slippage_bps": 0.0,
            "avg_slippage_abs": 0.0,
            "avg_latency_sec": 0.0,
            "fill_ratio": 0.0,
            "total_trades": 0,
        }

    return {
        "avg_slippage_bps": total_slip_bps / total_count,
        "avg_slippage_abs": total_slip_abs / total_count,
        "avg_latency_sec": total_lat / total_count,
        "fill_ratio": float(total_filled) / float(total_count),
        "total_trades": total_trades,
    }


def suggest_exec_reward_weights(alpha_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Given Execution Alpha dict (symbol → side → broker → stats),
    return a dict of suggested reward weights for ExecutionAwareReward.

    Heuristics:
      - If slippage is large (in bps or $), increase slippage penalty magnitude.
      - If latency is high, increase latency penalty magnitude.
      - If fill ratio is low, increase clamp_penalty and pred_block_penalty.
    """
    stats = _aggregate_global_stats(alpha_dict)

    avg_slip_bps = stats["avg_slippage_bps"]
    avg_slip_abs = stats["avg_slippage_abs"]
    avg_lat = stats["avg_latency_sec"]
    fill_ratio = stats["fill_ratio"]
    total_trades = stats["total_trades"]

    # Base weights close to what we used in Phase 76/81
    slippage_weight = -1.0
    spread_weight = -0.5
    latency_weight = -0.0005
    fill_prob_weight = 1.0
    clamp_penalty = -0.3
    pred_block_penalty = -2.0
    risk_block_penalty = -3.0

    # If we have very little data, don't overreact
    if total_trades < 10:
        return {
            "slippage_weight": slippage_weight,
            "spread_weight": spread_weight,
            "latency_weight": latency_weight,
            "fill_prob_weight": fill_prob_weight,
            "clamp_penalty": clamp_penalty,
            "pred_block_penalty": pred_block_penalty,
            "risk_block_penalty": risk_block_penalty,
        }

    # ---- Slippage-based tuning ----
    # avg_slip_bps around 0 is great; > 5 bps is mediocre; > 20 bps is bad
    if avg_slip_bps > 20.0:
        slippage_weight *= 2.5  # much stronger penalty
    elif avg_slip_bps > 10.0:
        slippage_weight *= 2.0
    elif avg_slip_bps > 5.0:
        slippage_weight *= 1.5
    elif avg_slip_bps < -5.0:
        # negative slippage (price improvement): we can relax penalty
        slippage_weight *= 0.7

    # ---- Latency-based tuning ----
    # avg_lat in seconds; > 2s is mildly bad, > 5s is worse
    if avg_lat > 5.0:
        latency_weight *= 3.0
    elif avg_lat > 2.0:
        latency_weight *= 2.0

    # ---- Fill-ratio tuning ----
    # If fill ratio is low, increase penalties for "blocked" or "clamped" cases
    if fill_ratio < 0.5:
        clamp_penalty *= 2.0
        pred_block_penalty *= 1.5
    elif fill_ratio < 0.75:
        clamp_penalty *= 1.5

    return {
        "slippage_weight": float(slippage_weight),
        "spread_weight": float(spread_weight),
        "latency_weight": float(latency_weight),
        "fill_prob_weight": float(fill_prob_weight),
        "clamp_penalty": float(clamp_penalty),
        "pred_block_penalty": float(pred_block_penalty),
        "risk_block_penalty": float(risk_block_penalty),
    }
