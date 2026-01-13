from __future__ import annotations
import os, json, shutil, hashlib, logging
from pathlib import Path
from datetime import datetime

try:
    from tools.telegram_alerts import notify
except ImportError:
    notify = lambda msg, **kw: print("[TELEGRAM]", msg)

logger = logging.getLogger("PolicyDeployer")
logging.basicConfig(level=logging.INFO)


class PolicyDeployer:
    """Deploys the latest LIVE policy bundle to the runtime directory."""

    def __init__(self, registry_path: str, target_dir: str):
        self.registry_path = Path(registry_path)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load_registry(self):
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("policies", [])

    # ------------------------------------------------------------------
    def find_latest_live(self, policies):
        live = [p for p in policies if p.get("status", "").upper() == "LIVE"]
        if not live:
            raise ValueError("No LIVE policy found in registry.")
        live.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        return live[0]

    # ------------------------------------------------------------------
    def compute_checksum(self, model_path: Path) -> str:
        """Compute SHA-256 short checksum for model file."""
        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]

    # ------------------------------------------------------------------
    def deploy(self):
        policies = self.load_registry()
        latest = self.find_latest_live(policies)
        name, version = latest["name"], latest["version"]
        ts = latest.get("timestamp")
        logger.info(f"🚀 Deploying {name} ({version})")

        src_dir = Path(f"data/policies/{name}_{version}")
        if not src_dir.exists():
            raise FileNotFoundError(f"Bundle not found: {src_dir}")

        # clear previous target
        for item in self.target_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # copy bundle
        bundle_target = self.target_dir / src_dir.name
        shutil.copytree(src_dir, bundle_target)
        logger.info(f"📦 Copied bundle → {bundle_target}")

        # compute checksum from model file if exists
        model_path = None
        for ext in (".pt", ".pkl", ".bin"):
            p = next(bundle_target.glob(f"**/*{ext}"), None)
            if p:
                model_path = p
                break

        if not model_path:
            logger.warning("⚠️ No model file found for checksum; metadata only.")
            checksum = "NA"
        else:
            checksum = self.compute_checksum(model_path)
            reg_sum = latest.get("checksum")
            if reg_sum and reg_sum != checksum:
                msg = f"Checksum mismatch! Registry={reg_sum}, computed={checksum}"
                logger.warning(msg)
                notify(f"⚠️ {msg}", kind="system")
            else:
                logger.info(f"🔢 Model checksum verified: {checksum}")

        # write manifest
        manifest = {
            "policy_name": name,
            "version": version,
            "checksum": checksum,
            "deployed_at": datetime.utcnow().isoformat(),
            "source_timestamp": ts,
        }
        with open(self.target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"✅ Deployment complete for {name} ({version})")
        notify(f"🧠 Policy deployed → {name} {version}\nChecksum: {checksum}", kind="system")
        return manifest
