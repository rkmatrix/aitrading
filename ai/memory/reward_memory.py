"""
reward_memory.py – Persists per-step rewards for analysis and retraining.
"""

from __future__ import annotations
import pickle
import os
from typing import List, Dict, Any
import logging
import time

logger = logging.getLogger("RewardMemory")


class RewardMemory:
    def __init__(self, path: str = "data/runtime/reward_memory.pkl", maxlen: int = 10_000):
        self.path = path
        self.maxlen = maxlen
        self.data: List[Dict[str, Any]] = []

        # Try load existing
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
                    logger.info(f"✅ Loaded RewardMemory ({len(self.data)} entries).")
            except Exception as e:
                logger.warning(f"⚠️ Could not load reward memory: {e}")
                self.data = []

    # ------------------------------------------------------------
    def push(self, item: Dict[str, Any]):
        """Append new reward record."""
        self.data.append(item)
        if len(self.data) > self.maxlen:
            self.data = self.data[-self.maxlen:]

    # ------------------------------------------------------------
    def save(self):
        """Persist rewards to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
        logger.info(f"💾 RewardMemory saved ({len(self.data)} records).")

    # ------------------------------------------------------------
    def last(self, n: int = 5):
        """Get last N reward entries."""
        return self.data[-n:]
