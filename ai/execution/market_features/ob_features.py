from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LOBSnapshot:
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volatility: float

def compute_features(s: LOBSnapshot) -> Dict[str, Any]:
    spread = max(s.ask - s.bid, 1e-6)
    mid = (s.ask + s.bid) / 2
    imbalance = (s.bid_size - s.ask_size) / max(s.bid_size + s.ask_size, 1)
    return {
        "spread": spread,
        "mid": mid,
        "imbalance": imbalance,
        "volatility": s.volatility,
    }
