from __future__ import annotations
import os, json, hashlib, logging
from pathlib import Path
from datetime import datetime

try:
    from tools.telegram_alerts import notify
except ImportError:
    notify = lambda msg, **kw: print("[TELEGRAM]", msg)

logger = logging.getLogger("PolicyLoader")
logging.basicConfig(level=logging.INFO)


class PolicyLoader:
    """
    Loads deployed runtime policy bundle.
    Validates manifest and optionally verifies checksum.
    """

    def __init__(self, manifest_path: str, strict_checksum: bool = True):
        self.manifest_path = Path(manifest_path)
        self.strict_checksum = strict_checksum
        self.policy_dir = self.manifest_path.parent

    # ------------------------------------------------------------------
    def _read_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def _verify_checksum(self, manifest: dict):
        """Compare file hash with manifest checksum."""
        expected = manifest.get("checksum")
        weights_path = next(self.policy_dir.glob("**/*.bin"), None)
        if not weights_path:
            logger.warning("No weight file found in current_policy bundle.")
            return True

        hasher = hashlib.sha256()
        with open(weights_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        actual = hasher.hexdigest()[:8]

        if actual != expected and self.strict_checksum:
            raise ValueError(
                f"Checksum mismatch: expected {expected}, got {actual}"
            )
        return True

    # ------------------------------------------------------------------
    def load_policy(self):
        """Main entry to load the deployed policy into memory."""
        manifest = self._read_manifest()
        name = manifest.get("policy_name")
        version = manifest.get("version")
        logger.info(f"🔍 Loading deployed policy {name} ({version})")

        try:
            self._verify_checksum(manifest)
        except Exception as e:
            notify(f"⚠️ Policy checksum failed: {e}", kind="system")
            if self.strict_checksum:
                raise

        # Load model weights or rule configuration
        model_path = next(self.policy_dir.glob("**/*.pt"), None)
        if not model_path:
            model_path = next(self.policy_dir.glob("**/*.pkl"), None)

        if not model_path:
            logger.warning("⚠️ No model file found. Loading metadata only.")
            return {"metadata": manifest, "model": None}

        # Placeholder for real model loading (torch / joblib)
        # In real system:
        # model = torch.load(model_path) or joblib.load(model_path)
        model = {"mock": "Loaded model data"}  # placeholder

        logger.info(f"✅ Policy {name} ({version}) loaded successfully.")
        return {"metadata": manifest, "model": model}
