"""
ai/policy/policy_loader.py
--------------------------
Loads trained or stub RL policy bundles for the AITradeBot.

This class reads a model manifest, handles version discovery,
and returns a lightweight policy interface with a `.predict()` stub
(if Stable-Baselines3/Torch are unavailable).

Used by: Phase 41 – Policy Sync, Phase 43 – Selector & Evolve,
Phase 47 – Rollback Guardian, Phase 48 – Recovery Trainer,
Phase 50 – Intelligent Executor.
"""

from __future__ import annotations
import os, json, logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# ======================================================
# ✅ Safe Torch / SB3 import (fallback if unavailable)
# ======================================================
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.envs import DummyVecEnv
except Exception as e:
    torch = None
    PPO = None
    DummyVecEnv = None
    logger.warning(f"⚠️  Torch/SB3 not available — running in STUB MODE ({e})")


# ======================================================
# 🧠 PolicyLoader
# ======================================================
class PolicyLoader:
    def __init__(self, root: str | Path, version: str = "latest"):
        """
        root: path to models/policies/<PolicyName>/
        version: folder name or 'latest'
        """
        self.root = Path(root)
        self.version = version
        self.bundle = None

    # --------------------------------------------------
    def _resolve_version(self) -> Path:
        if self.version == "latest":
            # Prefer a symlink or fall back to highest version number
            link = self.root / "latest"
            if link.exists():
                return Path(os.readlink(link)) if link.is_symlink() else link
            versions = sorted(
                [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v")],
                key=lambda p: p.name,
                reverse=True,
            )
            if not versions:
                raise FileNotFoundError(f"No version folders found under {self.root}")
            return versions[0]
        return self.root / self.version

    # --------------------------------------------------
    def _load_manifest(self, path: Path) -> dict:
        mpath = path / "manifest.json"
        if not mpath.exists():
            logger.warning(f"No manifest found in {path}, creating stub manifest.")
            return {"policy_name": path.name, "version": path.name}
        with open(mpath, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    def _load_model(self, path: Path):
        """Load PPO model if available, else a stub."""
        model_path = path / "model.pt"
        if not model_path.exists() or torch is None:
            logger.warning("⚠️  model.pt missing or Torch not available — using STUB policy.")
            return self._stub_policy()
        try:
            model = PPO.load(model_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            return self._stub_policy()

    # --------------------------------------------------
    def _stub_policy(self):
        """Return a minimal object with a predict() stub."""
        def _predict(obs):
            import random
            return random.choice(["BUY", "SELL", "HOLD"]), None
        return SimpleNamespace(predict=lambda obs: _predict(obs))

    # --------------------------------------------------
    def load(self):
        """Load policy manifest and model."""
        path = self._resolve_version()
        manifest = self._load_manifest(path)
        model = self._load_model(path)
        self.bundle = SimpleNamespace(path=path, manifest=manifest, model=model)
        logger.info(f"🌱 Loaded policy bundle {manifest.get('policy_name')} ({manifest.get('version')})")
        return self.bundle.model
