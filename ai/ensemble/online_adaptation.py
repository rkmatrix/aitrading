# ai/ensemble/online_adaptation.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque
import numpy as np


@dataclass
class BlendFeedback:
    reward: float
    drawdown: float
    volatility: float


class OnlineBlenderAdaptor:
    """Bandit-style adaptor that nudges blend weights using recent performance."""
    def __init__(self, window: int = 100, lr: float = 0.05):
        self.window = window
        self.lr = lr
        self.buffer: Deque[BlendFeedback] = deque(maxlen=window)
        self.last_weights: np.ndarray | None = None

    def push(self, fb: BlendFeedback) -> None:
        self.buffer.append(fb)

    def adapt(self, weights: np.ndarray) -> np.ndarray:
        if not self.buffer:
            return weights
        r = np.mean([b.reward for b in self.buffer])
        dd = np.mean([b.drawdown for b in self.buffer])
        adj = r - 0.5 * dd
        new_w = weights + self.lr * adj
        new_w = np.clip(new_w, 1e-6, None)
        new_w = new_w / new_w.sum()
        self.last_weights = new_w
        return new_w
