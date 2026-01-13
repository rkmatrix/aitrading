"""
Meta Feedback Ingestor
----------------------
Loads feedback logs written by OnlineLearner (data/feedback_store.parquet),
aggregates reward statistics, and updates the MetaController for adaptive
learning across actions, symbols, and reward conditions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


class MetaFeedbackIngestor:
    """
    Reads feedback_store.parquet, aggregates reward/action stats,
    and provides retraining data for MetaController or MetaMemory.
    """

    def __init__(self, data_dir="data", min_samples: int = 5):
        self.data_dir = Path(data_dir)
        self.feedback_path = self.data_dir / "feedback_store.parquet"
        self.min_samples = min_samples
        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def load_feedback(self) -> pd.DataFrame:
        """Load feedback data (returns empty DataFrame if missing)."""
        if not self.feedback_path.exists():
            return pd.DataFrame(columns=["timestamp", "action", "reward", "prob_0", "prob_1"])
        try:
            df = pd.read_parquet(self.feedback_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["reward"])
            return df
        except Exception as e:
            print(f"⚠️  Failed to load feedback store: {e}")
            return pd.DataFrame(columns=["timestamp", "action", "reward", "prob_0", "prob_1"])

    # ------------------------------------------------------------------
    def summarize(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return reward summary by action."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df = self.load_feedback()
        if df.empty:
            return pd.DataFrame(columns=["action", "count", "avg_reward", "success_rate"])

        summary = (
            df.groupby("action")
            .agg(
                count=("reward", "size"),
                avg_reward=("reward", "mean"),
                success_rate=("reward", lambda x: np.mean(np.array(x) > 0)),
            )
            .reset_index()
        )
        return summary.sort_values("avg_reward", ascending=False)

    # ------------------------------------------------------------------
    def get_training_batch(self, window: str = "7D") -> Optional[pd.DataFrame]:
        """
        Extracts a sliding window of feedback (e.g., last 7 days)
        to train or fine-tune MetaController.
        Handles tz-aware vs tz-naive timestamp mismatches.
        """
        df = self.load_feedback()
        if df.empty:
            return None
    
        # Normalize all timestamps to UTC and remove tz awareness
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(window)
        recent = df[df["timestamp"] >= cutoff]
    
        return recent if len(recent) >= self.min_samples else None


    # ------------------------------------------------------------------
    def export_csv(self, out_path: str = "data/meta_feedback_summary.csv") -> None:
        """Write reward summary to CSV for dashboard visualization."""
        summary = self.summarize()
        summary.to_csv(out_path, index=False)
        print(f"✅ Feedback summary exported → {out_path}")

    # ------------------------------------------------------------------
    def to_meta_payload(self) -> Dict[str, float]:
        """
        Convert the latest reward stats into a compact dict for
        MetaController parameter tuning (e.g., risk weighting).
        """
        df = self.load_feedback()
        if df.empty:
            return {"avg_reward": 0.0, "success_rate": 0.5}

        recent = self.get_training_batch(window="14D") or df
        avg_reward = recent["reward"].mean()
        success_rate = np.mean(recent["reward"] > 0)

        # example meta signal
        meta_payload = {
            "avg_reward": float(avg_reward),
            "success_rate": float(success_rate),
            "timestamp": datetime.utcnow().isoformat(),
        }
        return meta_payload
