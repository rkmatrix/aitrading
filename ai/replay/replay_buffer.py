# ai/replay/replay_buffer.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Sequence
from collections import deque
import random


@dataclass
class Transition:
    """
    Generic RL-like transition object for trading.

    For now this is intentionally simple and event-centric:
        ts, symbol, price, position, reward, equity, extras
    Later you can evolve this into full (state, action, next_state) tuples.
    """
    ts: float
    symbol: str
    price: float
    position: float
    reward: float
    equity: float
    extras: Dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = int(capacity)
        self._buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def extend(self, transitions: Sequence[Transition]) -> None:
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            batch_size = len(self._buffer)
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self._buffer]

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self),
            "capacity": self.capacity,
        }
