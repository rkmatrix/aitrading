from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderRequest:
    symbol: str
    side: str          # BUY | SELL
    qty: int
    order_type: str    # MARKET | LIMIT
    limit_price: Optional[float] = None
    tif: str = "DAY"   # DAY | IOC | FOK
    client_order_id: Optional[str] = None
    meta: Optional[dict] = None

@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    status: str
    venue: str
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    error: Optional[str] = None
