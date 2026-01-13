# ai/policy/auto_retrain_engine.py
"""
Phase 101 — Auto Retrain with Real OHLCV (Multi-Symbol) + Auto Phase 99/100
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO

from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify

from ai.env.env_registration import register_env
from ai.policy.leaderboard_engine import PolicyLeaderboardEngine
from ai.policy.promotion_engine import PolicyPromotionEngine

log = logging.getLogger(__name__)


@dataclass
class AutoRetrainConfig:
    env_id: str = "TradingEnv-v0"
    policy_root: str = "models/policies"
    out_root: str = "models/policies"
    base_policies: Optional[List[str]] = None
    total_timesteps: int = 100_000
    shadow_tag: str = "shadow"
    telegram_alerts: bool = True
    run_phase99_after: bool = True
    run_phase100_after: bool = True
    symbols: Optional[List[str]] = None  # <-- multi-symbol support


class AutoRetrainEngine:
    """
    Phase 101 — Real Data Retrain + Auto Leaderboard + Auto Promotion.

    We DO NOT load old PPO weights (they used different obs spaces).
    Instead we always start a fresh PPO model compatible with the
    current TradingEnv-v0 (multi-symbol OHLCV).
    """

    def __init__(self, cfg: AutoRetrainConfig) -> None:
        ensure_env_loaded()
        register_env()
        self.cfg = cfg

        self.policy_root = Path(cfg.policy_root)
        self.out_root = Path(cfg.out_root)

        if not self.policy_root.exists():
            raise FileNotFoundError(self.policy_root)

        if not self.cfg.base_policies:
            log.warning("No base_policies configured for AutoRetrainEngine.")

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _alert(self, msg: str, kind: str = "system", meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.telegram_alerts:
            return
        try:
            notify(msg, kind=kind, meta=meta or {})
        except Exception:
            pass

    def _read_manifest(self, name: str) -> Dict[str, Any]:
        p = self.policy_root / name / "manifest.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _shadow_name(self, base: str) -> str:
        t = time.strftime("%Y%m%d_%H%M%S")
        return f"{base}_{self.cfg.shadow_tag}_{t}"

    def _make_env(self):
        """
        Create the multi-symbol OHLCV environment.
        """
        env_kwargs: Dict[str, Any] = {}
        if self.cfg.symbols:
            env_kwargs["symbols"] = self.cfg.symbols

        return gym.make(self.cfg.env_id, **env_kwargs)

    def _write_shadow(
        self,
        base: str,
        shadow: str,
        manifest: Dict[str, Any],
        model: PPO,
    ) -> Path:
        out = self.out_root / shadow
        out.mkdir(parents=True, exist_ok=True)

        model.save(str(out / "model.zip"))

        m = dict(manifest)
        m["policy_name"] = shadow
        m["based_on"] = base
        m["shadow"] = True
        m["version"] = m.get("version", "v1.0.0")
        m["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.cfg.symbols:
            m["symbols"] = list(self.cfg.symbols)

        (out / "manifest.json").write_text(json.dumps(m, indent=2))
        return out

    # -----------------------------------------------------
    # Core
    # -----------------------------------------------------
    def retrain_one(self, pol: str) -> str:
        self._alert(f"🧠 Phase 101: starting retrain for base policy '{pol}'", "rl_decisions")

        manifest = self._read_manifest(pol)
        env = self._make_env()

        # Fresh PPO model (no old weights)
        # Load hyperparameters if available
        import json, os
        hp_path = Path("data/hparams/best_p101.json")
        if hp_path.exists():
            try:
                with open(hp_path, "r") as f:
                    hparams = json.load(f)
                # Remove non-PPO parameters
                hparams = {k: v for k, v in hparams.items() if k not in ["score", "final_equity", "drawdown"]}
            except:
                hparams = {}
        else:
            hparams = {}
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=None,
            **hparams,
        )

        model.learn(total_timesteps=self.cfg.total_timesteps)

        shadow = self._shadow_name(pol)
        out_dir = self._write_shadow(pol, shadow, manifest, model)

        self._alert(
            f"✅ Phase 101 retrain complete for '{pol}' → shadow='{shadow}'",
            "rl_decisions",
            meta={"base_policy": pol, "shadow_policy": shadow, "out_dir": str(out_dir)},
        )

        return shadow

    def _run_phase99(self) -> None:
        log.info("📊 Running Phase 99 Leaderboard...")
        engine = PolicyLeaderboardEngine()
        engine.run()

    def _run_phase100(self) -> None:
        log.info("🏆 Running Phase 100 Promotion Engine...")
        prom = PolicyPromotionEngine()
        prom.run()

    def run(self) -> None:
        if not self.cfg.base_policies:
            log.warning("AutoRetrainEngine.run(): no base_policies to retrain.")
            return

        for pol in self.cfg.base_policies:
            try:
                _ = self.retrain_one(pol)
            except Exception as e:
                log.error(f"Retrain failed for {pol}: {e}", exc_info=True)

        if self.cfg.run_phase99_after:
            self._run_phase99()

        if self.cfg.run_phase100_after:
            self._run_phase100()

        log.info("🎉 Phase 101 completed.")
