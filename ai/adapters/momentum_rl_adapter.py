# ai/adapters/momentum_rl_adapter.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from ai.adapters.base_policy_adapter import PolicyAdapter
from ai.models.momentum_rl import MomentumRL  # your RL model module

logger = logging.getLogger(__name__)

class MomentumRLAdapter(PolicyAdapter):
    """Adapter for MomentumRL policy."""
    def __init__(self):
        self.name = "MomentumRL"
        self.model = MomentumRL.load_or_init()
        logger.info("✅ %s adapter initialized", self.name)

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            out = self.model.predict(obs)
            targets = out.get("signals", {})
            conf = float(out.get("confidence", 0.8))
            return {"targets": targets, "confidence": conf, "meta": {"src": self.name}}
        except Exception as e:
            logger.exception("%s predict failed: %s", self.name, e)
            return {"targets": {}, "confidence": 0.0, "meta": {"error": str(e)}}

    def update(self, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.model.update_reward(reward)
        except Exception:
            pass
