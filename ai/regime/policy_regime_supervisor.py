# ai/regime/policy_regime_supervisor.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .regime_detector import MarketRegimeDetector

logger = logging.getLogger("PolicyRegimeSupervisor")


@dataclass
class PolicyRegimeConfig:
    regime_state_file: Path
    policy_mapping: Dict[str, Dict[str, Any]]
    policy_name: str = "EquityRLPolicy"


class PolicyRegimeSupervisor:
    """
    Phase 59 — Market Regime Supervisor

    - Runs MarketRegimeDetector
    - Maps regime → policy profile (aggressive/defensive/balanced)
    - Writes JSON file for other components to consume.
    """

    def __init__(self, cfg: Dict[str, Any]):
        paths = cfg.get("paths", {})
        self.cfg = PolicyRegimeConfig(
            regime_state_file=Path(paths.get("regime_state_file", "data/runtime/phase59_regime_state.json")),
            policy_mapping=cfg.get("policy_mapping", {}),
            policy_name=cfg.get("policy_name", "EquityRLPolicy"),
        )
        self.detector = MarketRegimeDetector(cfg)

    def run_once(self) -> Dict[str, Any]:
        """
        Detects regime and writes a regime_state JSON.
        Returns the state dict.
        """
        logger.info("🧭 Phase59: Running Market Regime Supervisor…")

        result = self.detector.detect_regime()
        mapping = self.cfg.policy_mapping.get(result.regime, {})

        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "policy_name": self.cfg.policy_name,
            "regime": result.regime,
            "profile": mapping.get("profile", "unknown"),
            "mapping_notes": mapping.get("notes", ""),
            "metrics": {
                "vol": result.vol,
                "slope": result.slope,
                **result.details,
            },
        }

        # Write to JSON for others to consume
        self.cfg.regime_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.regime_state_file.write_text(json.dumps(state, indent=2))
        logger.info("💾 Regime state written → %s", self.cfg.regime_state_file)

        return state
