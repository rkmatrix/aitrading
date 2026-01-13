from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    lot_size: int = 1
    tick_size: float = 0.01
    is_option: bool = False
    is_crypto: bool = False
    primary_exchange: Optional[str] = None
