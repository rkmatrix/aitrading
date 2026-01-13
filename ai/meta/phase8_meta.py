# ai/meta/phase8_meta.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import json, datetime as dt
from typing import Dict
from dotenv import load_dotenv       # ✅ new
load_dotenv()  

META_DIR = Path("data/meta")
META_DIR.mkdir(parents=True, exist_ok=True)

MEM_FILE = META_DIR / "performance_memory.csv"
WEIGHT_FILE = META_DIR / "adaptive_weights.csv"


class MetaLearner:
    """
    Learns from its own trade outcomes and adapts future allocation weights.
    """

    def __init__(self, decay: float = 0.9, learning_rate: float = 0.1):
        self.decay = decay
        self.lr = learning_rate
        self.weights = self._load_weights()

    # ----------------------------------------------------------
    # Load / save persistent weights
    # ----------------------------------------------------------
    def _load_weights(self) -> pd.DataFrame:
        if WEIGHT_FILE.exists():
            try:
                return pd.read_csv(WEIGHT_FILE)
            except Exception:
                pass
        return pd.DataFrame(columns=["symbol", "momentum_w", "meanrev_w", "vol_w"])

    def _save_weights(self):
        try:
            self.weights.to_csv(WEIGHT_FILE, index=False)
        except Exception:
            pass

    # ----------------------------------------------------------
    # Compute feedback based on returns
    # ----------------------------------------------------------
    def compute_feedback(self, allocs: Dict[str, float], returns: Dict[str, float]) -> Dict[str, float]:
        """
        Compare predicted allocations vs realized returns to create feedback signals.
        Positive feedback if direction & sign agree, negative otherwise.
        """
        feedback = {}
        for sym, alloc in allocs.items():
            ret = returns.get(sym, 0.0)
            score = np.sign(alloc) * ret
            feedback[sym] = score
        return feedback

    # ----------------------------------------------------------
    # Apply feedback to allocations
    # ----------------------------------------------------------
    def apply_feedback(self, allocs: Dict[str, float], feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights dynamically using meta-learning logic.
        """
        if not feedback:
            return allocs

        updated = {}
        for sym, w in allocs.items():
            fb = feedback.get(sym, 0.0)
            # scaled adjustment with sigmoid dampening
            delta = np.tanh(fb * 5) * self.lr
            new_w = max(0.0, w * (1 + delta))
            updated[sym] = new_w

        # normalize again
        total = sum(updated.values())
        if total > 0:
            updated = {s: w / total for s, w in updated.items()}

        return updated

    # ----------------------------------------------------------
    # Record feedback metrics for learning history
    # ----------------------------------------------------------
    def save_feedback(self, feedback: Dict[str, float], as_of: pd.Timestamp):
        if not feedback:
            return
        as_of = pd.Timestamp(as_of).tz_convert("UTC") if hasattr(as_of, "tzinfo") else pd.Timestamp(as_of, tz="UTC")
        rows = [{"date": as_of.isoformat(), "symbol": s, "feedback": v} for s, v in feedback.items()]

        df = pd.DataFrame(rows)
        if MEM_FILE.exists():
            old = pd.read_csv(MEM_FILE)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(MEM_FILE, index=False)


# ----------------------------------------------------------
# Self-test
# ----------------------------------------------------------
if __name__ == "__main__":
    meta = MetaLearner()
    allocs = {"AAPL": 0.3, "MSFT": 0.2, "SPY": 0.5}
    rets = {"AAPL": 0.02, "MSFT": -0.01, "SPY": 0.01}
    fb = meta.compute_feedback(allocs, rets)
    new_allocs = meta.apply_feedback(allocs, fb)
    print("Feedback:", fb)
    print("Adjusted:", new_allocs)
