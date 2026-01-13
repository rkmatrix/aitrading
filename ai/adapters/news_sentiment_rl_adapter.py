# ai/adapters/news_sentiment_rl_adapter.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from ai.adapters.base_policy_adapter import PolicyAdapter
from ai.models.news_sentiment_rl import NewsSentimentRL  # your sentiment RL model

logger = logging.getLogger(__name__)

class NewsSentimentRLAdapter(PolicyAdapter):
    """Adapter for NewsSentimentRL (sentiment-driven RL agent)."""
    def __init__(self):
        self.name = "NewsSentimentRL"
        self.model = NewsSentimentRL.load_or_init()
        logger.info("✅ %s adapter initialized", self.name)

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pred = self.model.predict(obs)
            targets = pred.get("targets", {})
            conf = float(pred.get("confidence", 0.75))
            return {"targets": targets, "confidence": conf, "meta": {"src": self.name}}
        except Exception as e:
            logger.exception("%s predict failed: %s", self.name, e)
            return {"targets": {}, "confidence": 0.0, "meta": {"error": str(e)}}

    def update(self, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.model.update_reward(reward)
        except Exception:
            pass
