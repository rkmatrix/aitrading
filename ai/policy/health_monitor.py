from __future__ import annotations
import os, json, time, hashlib, logging
from pathlib import Path
from datetime import datetime, timedelta

try:
    from tools.telegram_alerts import notify
except ImportError:
    notify = lambda msg, **kw: print("[TELEGRAM]", msg)

logger = logging.getLogger("PolicyHealthMonitor")
logging.basicConfig(level=logging.INFO)


class PolicyHealthMonitor:
    """Continuously validates deployed policy bundle integrity and age."""

    def __init__(
        self,
        manifest_path: str,
        check_interval: int = 300,
        max_age_days: int = 3,
        alert_on_stale: bool = True,
        alert_on_corruption: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.check_interval = check_interval
        self.max_age_days = max_age_days
        self.alert_on_stale = alert_on_stale
        self.alert_on_corruption = alert_on_corruption

    # ------------------------------------------------------------------
    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        return json.load(open(self.manifest_path, "r", encoding="utf-8"))

    # ------------------------------------------------------------------
    def _compute_checksum(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]

    # ------------------------------------------------------------------
    def _find_model_file(self, bundle_dir: Path):
        for ext in (".pt", ".pkl", ".bin"):
            f = next(bundle_dir.glob(f"**/*{ext}"), None)
            if f:
                return f
        return None

    # ------------------------------------------------------------------
    def check_once(self):
        """Perform one-time integrity + age validation."""
        manifest = self._read_manifest()
        name, version = manifest["policy_name"], manifest["version"]
        logger.info(f"🔍 Checking health of {name} ({version})")

        bundle_dir = self.manifest_path.parent / f"{name}_{version}"
        if not bundle_dir.exists():
            msg = f"❌ Bundle folder missing: {bundle_dir}"
            logger.error(msg)
            if self.alert_on_corruption:
                notify(msg, kind="system")
            return False

        model_path = self._find_model_file(bundle_dir)
        if not model_path:
            msg = f"❌ No model file found in {bundle_dir}"
            logger.error(msg)
            if self.alert_on_corruption:
                notify(msg, kind="system")
            return False

        # checksum
        expected = manifest.get("checksum")
        actual = self._compute_checksum(model_path)
        if expected != actual:
            msg = f"⚠️ Checksum mismatch! expected={expected}, got={actual}"
            logger.warning(msg)
            if self.alert_on_corruption:
                notify(msg, kind="system")
        else:
            logger.info("✅ Checksum verified OK")

        # staleness
        deployed_at = datetime.fromisoformat(manifest.get("deployed_at"))
        age = (datetime.utcnow() - deployed_at).days
        if age > self.max_age_days:
            msg = f"⚠️ Policy {name} ({version}) is {age} days old (> {self.max_age_days})"
            logger.warning(msg)
            if self.alert_on_stale:
                notify(msg, kind="system")
        else:
            logger.info(f"🕒 Policy age {age} day(s) OK")
        return True

    # ------------------------------------------------------------------
    def run_forever(self):
        logger.info("🚀 Policy Health Monitor started.")
        while True:
            try:
                self.check_once()
            except Exception as e:
                msg = f"💥 Health check error: {e}"
                logger.error(msg)
                notify(msg, kind="system")
            time.sleep(self.check_interval)
