from __future__ import annotations
import json, csv, logging
from pathlib import Path
from datetime import datetime

try:
    from tools.telegram_alerts import notify
except ImportError:
    notify = lambda msg, **kw: print("[TELEGRAM]", msg)

logger = logging.getLogger("PolicyFeedback")
logging.basicConfig(level=logging.INFO)


class PolicyFeedbackLoop:
    """Stores live trading results into the policy registry and optional CSV."""

    def __init__(self, registry_path: str, manifest_path: str, feedback_path: str | None = None):
        self.registry_path = Path(registry_path)
        self.manifest_path = Path(manifest_path)
        self.feedback_path = Path(feedback_path) if feedback_path else None

    # ------------------------------------------------------------------
    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        return json.load(open(self.manifest_path, "r", encoding="utf-8"))

    # ------------------------------------------------------------------
    def _load_registry(self):
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        return json.load(open(self.registry_path, "r", encoding="utf-8"))

    # ------------------------------------------------------------------
    def _save_registry(self, data):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    def record_feedback(self, metrics: dict):
        """Add feedback metrics for currently deployed policy."""
        manifest = self._load_manifest()
        registry = self._load_registry()

        name, version = manifest["policy_name"], manifest["version"]
        key = (name, version)

        # locate matching registry entry
        for p in registry["policies"]:
            if p["name"] == name and p["version"] == version:
                if "feedback" not in p:
                    p["feedback"] = []
                feedback_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": metrics,
                }
                p["feedback"].append(feedback_entry)
                p["last_score"] = metrics.get("score", None)
                break
        else:
            logger.warning(f"No registry entry found for {key}")
            return

        self._save_registry(registry)
        logger.info(f"🧾 Feedback recorded for {name} ({version})")

        # optional CSV log
        if self.feedback_path:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.feedback_path.exists()
            with open(self.feedback_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "policy_name", "version", *metrics.keys()],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {"timestamp": datetime.utcnow().isoformat(),
                     "policy_name": name, "version": version, **metrics}
                )
            logger.info(f"📈 CSV feedback logged → {self.feedback_path}")

        notify(f"📊 Policy feedback logged → {name} {version}\nMetrics: {metrics}", kind="system")
