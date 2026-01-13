# ai/online/alpha_online_swap.py
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class AlphaOnlineSwap:
    """
    AlphaOnlineSwap

    - Performs a backed-up, nearly-atomic swap of PPO model bundle.
    - Optionally updates manifest.json with online-learning metadata.
    """

    def __init__(
        self,
        *,
        bundle_dir: str,
        model_file: str = "model.zip",
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.model_path = self.bundle_dir / model_file
        self.manifest_path = self.bundle_dir / "manifest.json"

        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Policy bundle directory not found: {self.bundle_dir}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def _update_manifest(self) -> None:
        if not self.manifest_path.exists():
            logger.info("AlphaOnlineSwap: no manifest.json found, skipping manifest update.")
            return

        try:
            with self.manifest_path.open("r", encoding="utf-8") as f:
                data: Dict = json.load(f)
        except Exception as e:
            logger.warning("AlphaOnlineSwap: failed to read manifest.json (%s), skipping: %s", self.manifest_path, e)
            return

        ts = int(time.time())
        data["last_online_update_ts"] = ts

        # bump a string version if present
        v = data.get("version")
        if isinstance(v, str):
            if "+online" not in v:
                data["version"] = v + "+online"
        else:
            data["version"] = "online"

        try:
            with self.manifest_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("AlphaOnlineSwap: manifest.json updated with online-learning metadata.")
        except Exception as e:
            logger.warning("AlphaOnlineSwap: failed to write manifest.json: %s", e)

    def swap(self, tmp_model_path: str | Path) -> None:
        """
        Swap the current model with tmp_model_path, keeping a backup.
        """
        tmp_model_path = Path(tmp_model_path)
        if not tmp_model_path.exists():
            raise FileNotFoundError(f"Temporary model file not found: {tmp_model_path}")

        backup_path = self.model_path.with_suffix(".backup.zip")

        logger.info("📦 AlphaOnlineSwap: backing up current model to %s", backup_path)
        shutil.copy2(self.model_path, backup_path)

        logger.info("🔁 AlphaOnlineSwap: swapping %s → %s", tmp_model_path, self.model_path)
        shutil.move(str(tmp_model_path), str(self.model_path))

        self._update_manifest()
        logger.info("✅ AlphaOnlineSwap: model swap complete.")
