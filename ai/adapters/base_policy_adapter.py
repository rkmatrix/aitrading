# ai/adapters/base_policy_adapter.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Protocol

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class PolicyAdapter(Protocol):
    name: str

    def predict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def update(self, reward: float, info: Optional[Dict[str, Any]] = None) -> None:
        ...
