# ai/ensemble/blender.py
from __future__ import annotations
import numpy as np
from typing import Any, List


class BlendEngine:
    """Combines actions from multiple policies using provided weights."""
    def __init__(self, action_mode: str = "continuous"):
        self.action_mode = action_mode

    def blend(self, actions: List[Any], weights: np.ndarray) -> Any:
        if len(actions) == 0:
            raise ValueError("No actions to blend")
        if len(actions) == 1:
            return actions[0]

        if self.action_mode == "continuous":
            arr = np.stack([np.array(a) for a in actions], axis=0)
            w = weights.reshape(-1, 1)
            return (arr * w).sum(axis=0)
        elif self.action_mode == "discrete":
            idx = int(np.argmax(weights))
            return actions[idx]
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
