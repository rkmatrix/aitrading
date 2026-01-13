"""
replay_buffer.py – Simple circular replay memory.
Used for short-term state/action/reward storage during training.
"""

from __future__ import annotations
from collections import deque
import random
from typing import Deque, Dict, Any, List


class ReplayBuffer:
    def __init__(self, maxlen: int = 10_000):
        self.maxlen = maxlen
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=maxlen)

    # ------------------------------------------------------------
    def push(self, experience: Dict[str, Any]):
        """Add new experience to buffer."""
        self.buffer.append(experience)

    # ------------------------------------------------------------
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Random batch sample."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.buffer)
