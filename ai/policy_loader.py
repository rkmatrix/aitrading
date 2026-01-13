# ai/policy_loader.py
"""
Generic PolicyLoader for AITradeBot
-----------------------------------
Used by all adapters (EquityRLPolicy, MomentumRL, etc.)
to fetch trained model bundles or lightweight mocks.

It looks for models under:
    models/policies/{policy_name}/{version}/manifest.json
If not found, it returns a DummyPolicy stub that responds to predict() safely.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------
# Dummy fallback for missing models
# ---------------------------------------------------------------------
class DummyPolicy:
    def __init__(self, name: str, version: str = "unknown"):
        self.name = name
        self.version = version
        logger.warning("⚠️ Using DummyPolicy for %s (%s). No model bundle found.", name, version)

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return random/neutral outputs for testing."""
        import random
        symbols = ["AAPL", "MSFT", "TSLA"]
        return {
            "targets": {s: random.uniform(-0.2, 0.2) for s in symbols},
            "confidence": 0.8,
            "meta": {"src": f"DummyPolicy-{self.name}", "version": self.version},
        }

    def update_reward(self, reward: float, info: Dict[str, Any] | None = None) -> None:
        pass


# ---------------------------------------------------------------------
# Real loader
# ---------------------------------------------------------------------
class PolicyLoader:
    ROOT = Path("models/policies")

    @classmethod
    def load(cls, policy_name: str, version: str = "latest") -> Any:
        """
        Try loading a policy from models/policies/{name}/{version}/manifest.json
        Fallback: DummyPolicy if not found.
        """
        policy_dir = cls.ROOT / policy_name / version
        manifest_path = policy_dir / "manifest.json"

        if not manifest_path.exists():
            logger.warning("⚠️ No manifest found for %s (%s). Returning DummyPolicy.", policy_name, version)
            return DummyPolicy(policy_name, version)

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("📦 Loaded policy manifest for %s (%s): %s", policy_name, version, meta.get("summary", "ok"))
        except Exception as e:
            logger.error("💥 Failed to read manifest for %s (%s): %s", policy_name, version, e)
            return DummyPolicy(policy_name, version)

        # Try loading model.pt / weights if defined
        model_path = policy_dir / "model.pt"
        if model_path.exists():
            logger.info("✅ Found model weights for %s (%s): %s", policy_name, version, model_path)
            # stub: replace this with your real torch model load if needed
            return DummyPolicy(policy_name, version)

        logger.warning("⚠️ No model.pt found for %s (%s). Using DummyPolicy fallback.", policy_name, version)
        return DummyPolicy(policy_name, version)
