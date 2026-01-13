# ai/policy/policy_promotion.py

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("ai.policy.policy_promotion")


class PolicyPromotionEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        paths = cfg.get("paths", {})
        self.root = Path(paths.get("policies_root", "models/policies/EquityRLPolicy"))
        self.metric = cfg.get("metric_to_compare", "sharpe")
        self.threshold = float(cfg.get("promotion_threshold", 0.03))
        self.min_episodes = int(cfg.get("min_episodes", 10))

    # -----------------------------
    # Helpers
    # -----------------------------
    def _load_metrics(self, dir_path: Path) -> Optional[Dict[str, Any]]:
        fp = dir_path / "metrics.json"
        if not fp.exists():
            return None
        try:
            return json.loads(fp.read_text())
        except Exception:
            logger.warning("Failed to parse metrics.json in %s", dir_path)
            return None

    def _get_latest_version_dir(self) -> Optional[Path]:
        candidates = [
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name != "current_policy"
        ]
        if not candidates:
            return None

        def parse(v: str):
            parts = v[1:].split(".")
            return tuple(map(int, parts))

        latest = max(candidates, key=lambda p: parse(p.name))
        return latest

    def _get_current_policy_dir(self) -> Path:
        return self.root / "current_policy"

    # -----------------------------
    # Promotion Logic
    # -----------------------------
    def _is_better(self, old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> bool:
        # If no old metrics → always promote
        if old is None:
            logger.info("No existing current_policy metrics — promoting new version by default.")
            return True

        episodes = new.get("episodes", 0)
        if episodes < self.min_episodes:
            logger.info(
                "New policy episodes (%d) < min_episodes (%d) → skip promotion",
                episodes, self.min_episodes,
            )
            return False

        old_val = float(old.get(self.metric, 0.0))
        new_val = float(new.get(self.metric, 0.0))
        target = old_val * (1.0 + self.threshold)

        logger.info(
            "Comparing '%s' metric: old=%.6f new=%.6f target=%.6f",
            self.metric, old_val, new_val, target,
        )

        if new_val <= target:
            logger.info("New metric did not beat threshold → no promotion.")
            return False

        # Optional guardrails: drawdown & winrate
        old_dd = float(old.get("max_dd", 0.0))
        new_dd = float(new.get("max_dd", 0.0))
        old_wr = float(old.get("winrate", 0.0))
        new_wr = float(new.get("winrate", 0.0))

        if new_dd < old_dd - 0.10:
            logger.info(
                "New max_dd (%.4f) much worse than old (%.4f) → no promotion.",
                new_dd, old_dd,
            )
            return False

        if new_wr < old_wr:
            logger.info(
                "New winrate (%.4f) worse than old (%.4f) → no promotion.",
                new_wr, old_wr,
            )
            return False

        return True

    def _promote(self, latest_dir: Path) -> bool:
        current_dir = self._get_current_policy_dir()

        if current_dir.exists():
            shutil.rmtree(current_dir)
        shutil.copytree(latest_dir, current_dir)

        logger.info("🏆 PROMOTED → %s → current_policy", latest_dir.name)
        return True

    # -----------------------------
    # Public entry
    # -----------------------------
    def promote(self) -> bool:
        latest_dir = self._get_latest_version_dir()
        if latest_dir is None:
            logger.error("❌ No versioned policy folders found under %s", self.root)
            return False

        new_metrics = self._load_metrics(latest_dir)
        if new_metrics is None:
            logger.error("❌ Latest version %s missing metrics.json.", latest_dir)
            return False

        current_dir = self._get_current_policy_dir()
        old_metrics = self._load_metrics(current_dir) if current_dir.exists() else None

        if not self._is_better(old_metrics, new_metrics):
            logger.info("🔒 Policy promotion skipped — criteria not satisfied.")
            return False

        return self._promote(latest_dir)
