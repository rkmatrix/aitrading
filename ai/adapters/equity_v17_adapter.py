# ai/adapters/equity_v17_adapter.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from ai.adapters.base_policy_adapter import PolicyAdapter
from ai.policy_loader import PolicyLoader  # your existing loader

logger = logging.getLogger(__name__)

class EquityV17Adapter(PolicyAdapter):
    """Adapter wrapper for EquityRLPolicy v1.7.3"""
    def __init__(self):
        self.name = "EquityRLPolicy v1.7.3"
        self._policy = PolicyLoader.load("EquityRLPolicy", version="1.7.3")
        logger.info("✅ %s adapter initialized", self.name)

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = self._policy.predict(obs)
            targets = raw.get("targets", {})
            conf = float(raw.get("confidence", 0.9))
            return {"targets": targets, "confidence": conf, "meta": {"src": self.name}}
        except Exception as e:
            logger.exception("%s prediction error: %s", self.name, e)
            return {"targets": {}, "confidence": 0.0, "meta": {"error": str(e)}}

    def update(self, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        try:
            if hasattr(self._policy, "update_reward"):
                self._policy.update_reward(reward, info)
        except Exception as e:
            logger.warning("%s update() failed: %s", self.name, e)
