from __future__ import annotations
from typing import Dict, Any

def apply_guardian_mode(mode: str) -> str:
    """
    Try to set the Guardian risk mode. If your project doesn't have it, this is a no-op.
    """
    try:
        from ai.guardian.core import set_mode  # hypothetical
        set_mode(mode)
        return "guardian:set_mode ok"
    except Exception:
        # no guardian present or different API; silently skip
        return "guardian:no-op"

def apply_router_weights(weights: Dict[str, float]) -> str:
    """
    Try to update SmartOrderRouter weighting hints. If not present, no-op.
    """
    try:
        from ai.execution.smart_router import set_weights  # hypothetical
        set_weights(weights)
        return "router:set_weights ok"
    except Exception:
        return "router:no-op"

def apply_throttle(order_rate_per_min: int) -> str:
    """
    Try to set a global throttle on order submissions.
    """
    try:
        from ai.execution.rate_limiter import set_order_rate_limit  # hypothetical
        set_order_rate_limit(order_rate_per_min)
        return "ratelimit:set ok"
    except Exception:
        return "ratelimit:no-op"

def apply_position_caps(per_symbol_caps: Dict[str, float]) -> str:
    """
    Try to set per-symbol caps in your allocator/guardian.
    """
    try:
        from ai.risk.position_caps import set_caps  # hypothetical
        set_caps(per_symbol_caps)
        return "caps:set ok"
    except Exception:
        return "caps:no-op"
