from dataclasses import dataclass

@dataclass
class ExecDecision:
    style: str
    params: dict

def choose_style(default_style: str, signal_strength: float, est_liquidity: float) -> ExecDecision:
    """
    Very simple policy:
    - strong signals → POV (more aggressive)
    - low liquidity → SMART_SLICE (post-only/patient)
    - otherwise → VWAP
    """
    if signal_strength is None:
        return ExecDecision(style=default_style, params={})
    s = float(signal_strength)
    if s >= 0.75 or s <= -0.75:
        return ExecDecision(style="POV", params={})
    if est_liquidity < 0.2:
        return ExecDecision(style="SMART_SLICE", params={"post_only": True})
    return ExecDecision(style="VWAP", params={})
