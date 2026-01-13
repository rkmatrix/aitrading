from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class Feature:
    name: str
    value: float
    ts: datetime
    src: str

@dataclass
class Signal:
    symbol: str
    side: str            # "BUY" | "SELL" | "FLAT"
    strength: float      # -1..1
    size: int            # units (shares)
    reason: Optional[str] = None
    meta: Optional[Dict] = None
