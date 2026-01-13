from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Event:
    kind: str                # "decision" | "fill" | "cancel" | "risk"
    payload: Dict[str, Any]

class FeedbackBus:
    def __init__(self, maxlen: int = 5000):
        self.maxlen = maxlen
        self._buf: List[Event] = []

    def publish(self, kind: str, payload: Dict[str, Any]):
        self._buf.append(Event(kind=kind, payload=payload))
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    def latest(self, n: int = 50) -> List[Event]:
        return self._buf[-n:]

    def snapshot_info(self) -> Dict[str, Any]:
        # Example: derive slippage stats or trade counts
        trades = [e for e in self._buf if e.kind in ("fill", "decision")]
        slippages = [e.payload.get("slippage_bps", 0.0) for e in self._buf if e.kind == "fill"]
        return {
            "trades_last": len(trades),
            "avg_slippage_bps": (sum(slippages) / len(slippages)) if slippages else 0.0
        }
