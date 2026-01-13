# ai/monitoring/continual_improvement.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("Phase58Continual")


@dataclass
class ContinualConfig:
    policies_root: Path
    replay_file: Path
    state_file: Path
    min_new_replay_rows: int
    max_policy_age_days: int
    policy_name: str = "EquityRLPolicy"


class ContinualImprovementCoordinator:
    """
    Phase 58 — Continual Policy Improvement Loop.

    Responsibilities:
      - Inspect replay file
      - Check when last retrain happened & how many rows used
      - Decide if retrain is needed
      - If yes: call Phase56 (train) and Phase57 (promotion)
      - Update state file
    """

    def __init__(self, cfg: Dict[str, Any]):
        paths = cfg.get("paths", {})
        retrain = cfg.get("retrain", {})

        self.cfg = ContinualConfig(
            policies_root=Path(paths.get("policies_root", "models/policies/EquityRLPolicy")),
            replay_file=Path(paths.get("replay_file", "data/replay/phase55_replay.jsonl")),
            state_file=Path(paths.get("state_file", "data/runtime/phase58_state.json")),
            min_new_replay_rows=int(retrain.get("min_new_replay_rows", 20)),
            max_policy_age_days=int(retrain.get("max_policy_age_days", 7)),
            policy_name=cfg.get("policy_name", "EquityRLPolicy"),
        )

    # -----------------------------
    # State handling
    # -----------------------------
    def _load_state(self) -> Dict[str, Any]:
        if not self.cfg.state_file.exists():
            return {
                "last_retrain_at": None,
                "last_replay_rows": 0,
                "last_version": None,
            }
        try:
            return json.loads(self.cfg.state_file.read_text())
        except Exception:
            logger.warning("Failed to parse state file; starting fresh.")
            return {
                "last_retrain_at": None,
                "last_replay_rows": 0,
                "last_version": None,
            }

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.state_file.write_text(json.dumps(state, indent=2))
        logger.info("💾 Phase58 state saved → %s", self.cfg.state_file)

    # -----------------------------
    # Replay & policy info
    # -----------------------------
    def _count_replay_rows(self) -> int:
        if not self.cfg.replay_file.exists():
            logger.warning("Replay file does not exist: %s", self.cfg.replay_file)
            return 0

        count = 0
        with self.cfg.replay_file.open("r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        logger.info("📦 Replay rows present: %d", count)
        return count

    def _get_latest_version_dir(self) -> Optional[Path]:
        root = self.cfg.policies_root
        if not root.exists():
            return None

        candidates = [
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name != "current_policy"
        ]
        if not candidates:
            return None

        def parse(v: str):
            parts = v[1:].split(".")
            return tuple(map(int, parts))

        latest = max(candidates, key=lambda p: parse(p.name))
        return latest

    def _get_current_policy_dir(self) -> Optional[Path]:
        cp = self.cfg.policies_root / "current_policy"
        return cp if cp.exists() and cp.is_dir() else None

    def _get_policy_age_days(self) -> Optional[int]:
        """
        Reads current_policy/manifest.json 'created_at' or file mtime.
        """
        cp = self._get_current_policy_dir()
        if cp is None:
            return None

        manifest_path = cp / "manifest.json"
        created_at = None

        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                created_at = m.get("created_at")
            except Exception:
                created_at = None

        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", ""))
            except Exception:
                dt = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        else:
            dt = datetime.fromtimestamp(manifest_path.stat().st_mtime)

        age_days = (datetime.utcnow() - dt).days
        return age_days

    # -----------------------------
    # Decision logic
    # -----------------------------
    def _should_retrain(self, state: Dict[str, Any], replay_rows: int) -> (bool, str):
        last_replay_rows = int(state.get("last_replay_rows", 0))
        delta_rows = max(0, replay_rows - last_replay_rows)

        logger.info(
            "🔍 Replay rows since last retrain: %d (total=%d, last_used=%d)",
            delta_rows, replay_rows, last_replay_rows,
        )

        if delta_rows < self.cfg.min_new_replay_rows:
            return False, f"Only {delta_rows} new rows (< {self.cfg.min_new_replay_rows} threshold)."

        # Optional policy age condition
        if self.cfg.max_policy_age_days > 0:
            age_days = self._get_policy_age_days()
            if age_days is not None and age_days < self.cfg.max_policy_age_days:
                return False, f"Current policy age {age_days}d < {self.cfg.max_policy_age_days}d threshold."

        return True, "New replay rows threshold satisfied."

    # -----------------------------
    # External phase calls
    # -----------------------------
    def _run_phase56(self) -> None:
        """
        Call Phase56 training runner programmatically.
        """
        from runner.phase56_train_replay import main as phase56_main

        logger.info("🚀 [Phase58] Triggering Phase56 replay training…")
        phase56_main()
        logger.info("✅ Phase56 replay training completed from Phase58.")

    def _run_phase57(self) -> None:
        """
        Call Phase57 promotion runner programmatically.
        """
        from runner.phase57_policy_promotion import main as phase57_main

        logger.info("🚀 [Phase58] Triggering Phase57 policy promotion…")
        phase57_main()
        logger.info("✅ Phase57 promotion completed from Phase58.")

    # -----------------------------
    # Public entry
    # -----------------------------
    def run_once(self) -> bool:
        """
        Runs a single check → maybe retrain → maybe promote → update state.
        Returns True if a retrain was performed, else False.
        """
        logger.info("🔁 Phase58 Continual Improvement check starting…")

        state = self._load_state()
        replay_rows = self._count_replay_rows()

        if replay_rows == 0:
            logger.info("⚠️ No replay rows available; skipping retrain.")
            return False

        should, reason = self._should_retrain(state, replay_rows)
        if not should:
            logger.info("⏸ No retrain needed: %s", reason)
            return False

        logger.info("✅ Retrain condition met: %s", reason)

        # Run Phase56 and Phase57
        self._run_phase56()
        self._run_phase57()

        # Determine latest version after promotion
        latest = self._get_latest_version_dir()
        latest_name = latest.name if latest is not None else None

        # Update state
        state["last_retrain_at"] = datetime.utcnow().isoformat()
        state["last_replay_rows"] = replay_rows
        state["last_version"] = latest_name

        self._save_state(state)

        logger.info("🎉 Phase58 retrain + promotion cycle completed.")
        return True
