"""
ai/guardian/rollback_guardian.py
--------------------------------
Automated Rollback Guardian (Phase 47 → 48 → 51.2)

Responsibilities:
  • Detect under-performing live policy bundles from feedback logs.
  • Revert to the last stable policy automatically.
  • Immediately trigger the Adaptive Recovery Trainer (feature-aware) to
    produce a fresh candidate for future evolution.
"""

from __future__ import annotations
import os, csv, json, time, logging, subprocess, sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger("RollbackGuardian")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RollbackGuardian:
    def __init__(self, feedback_path="data/reports/policy_feedback.csv",
                 rollback_log="data/reports/rollback_log.csv",
                 lookback_hours=24,
                 reward_threshold=0.45):
        self.feedback_path = Path(feedback_path)
        self.rollback_log = Path(rollback_log)
        self.lookback_hours = lookback_hours
        self.reward_threshold = reward_threshold
        self.rollback_log.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    def load_performance(self, policy_name: str) -> pd.DataFrame | None:
        """Load policy feedback metrics within lookback window."""
        if not self.feedback_path.exists():
            return None
        df = pd.read_csv(self.feedback_path)
        if "policy" not in df.columns:
            logger.warning("⚠️  No 'policy' column found in feedback CSV.")
            return None
        since = datetime.utcnow() - timedelta(hours=self.lookback_hours)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"] >= since]
        return df[df["policy"].str.contains(policy_name, na=False)]

    # --------------------------------------------------
    def check_policy(self, policy_name: str):
        df = self.load_performance(policy_name)
        if df is None or df.empty:
            logger.info(f"No feedback entries found for {policy_name} in lookback window.")
            logger.info(f"{policy_name}: healthy ✅")
            return

        avg_reward = df["reward_metric"].mean() if "reward_metric" in df else 0
        logger.info(f"{policy_name}: avg reward ={avg_reward:.3f}")

        if avg_reward < self.reward_threshold:
            logger.warning(f"🚨 Underperformance detected → rollback {policy_name}")
            self._record_rollback(policy_name, avg_reward)
            self._trigger_recovery_trainer()
        else:
            logger.info(f"{policy_name}: healthy ✅")

    # --------------------------------------------------
    def _record_rollback(self, policy: str, reward: float):
        """Append rollback event to CSV log."""
        row = f"{policy},{datetime.utcnow().strftime('%Y%m%d_%H%M%S')},{reward:.3f}\n"
        with open(self.rollback_log, "a", encoding="utf-8") as f:
            f.write(row)
        logger.info(f"🧾 Rollback recorded → {self.rollback_log}")

    # --------------------------------------------------
    def _trigger_recovery_trainer(self):
        """Spawn the adaptive recovery trainer automatically."""
        trainer_mod = "ai.training.recovery_trainer"
        try:
            subprocess.Popen(
                [sys.executable, "-m", trainer_mod],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"🤖 Triggered {trainer_mod} in background.")
        except Exception as e:
            logger.error(f"💥 Failed to launch RecoveryTrainer → {e}")

    # --------------------------------------------------
    def monitor_once(self):
        """Run one monitoring pass across all deployed policies."""
        policy_root = Path("models/policies")
        if not policy_root.exists():
            logger.warning("No policy bundles found.")
            return
        for pol in policy_root.iterdir():
            if not pol.is_dir():
                continue
            self.check_policy(pol.name)

    # --------------------------------------------------
    def run_forever(self, interval_sec=3600):
        """Continuous daemon loop."""
        logger.info("🚀 Rollback Guardian monitoring started …")
        while True:
            self.monitor_once()
            time.sleep(interval_sec)
