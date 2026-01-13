from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from time import time

@dataclass
class Order:
    symbol: str
    side: str                # "BUY" | "SELL"
    qty: float               # shares (positive)
    order_type: str = "MARKET"
    time_in_force: str = "day"
    tag: str = "phase35"
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderResult:
    ok: bool
    broker: str
    order_id: Optional[str]
    status: str
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time)
    error: Optional[str] = None
