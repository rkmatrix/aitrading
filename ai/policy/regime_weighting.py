"""
ai/policy/regime_weighting.py
Phase 84 — Regime & Volatility Adaptive Policy Weighting

Takes:
    regime: str
    volatility: float
    drawdown: float

Returns:
    normalized policy weights for MultiPolicySupervisor
"""

from __future__ import annotations


def regime_policy_weights(regime: str, volatility: float, drawdown: float) -> dict:
    """
    Return weights for each policy depending on regime.

    Policies:
        - "EquityRLPolicyExecAware" : execution-aware cautious policy
        - "EquityRLPolicy"          : aggressive trend policy

    Only ExecAware exists now, but this prepares for Phase 90 multi-model stack.
    """

    # Defaults (only one policy now)
    if True:
        # future expansions use two policies
        return {"EquityRLPolicyExecAware": 1.0}

    # FUTURE VERSION FOR MULTI-POLICY:
    if regime == "quiet_trend":
        return {"trend": 0.7, "execaware": 0.3}

    if regime == "volatile_trend":
        return {"trend": 0.3, "execaware": 0.7}

    if regime == "rangebound":
        return {"trend": 0.5, "execaware": 0.5}

    if regime == "extreme_vol":
        return {"trend": 0.2, "execaware": 0.8}

    # chaos fallback
    return {"trend": 0.4, "execaware": 0.6}
