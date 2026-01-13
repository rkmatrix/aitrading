"""
ai/policy/agents/ml_alphazoo_agent.py

Compatibility stub for FusionBrain (Phase 92).
AlphaZoo Agents become available only in Phase 102 and above.

This lightweight class keeps FusionBrain functional even when
no AlphaZoo models are deployed.
"""

from __future__ import annotations


class DummyMLAlphaZooAgent:
    """
    A harmless no-op agent.
    FusionBrain expects this class to have:
        - load(path)
        - predict(features)
        - get_info()
    """

    def __init__(self, *args, **kwargs):
        self.loaded = False

    @staticmethod
    def load(path: str) -> "DummyMLAlphaZooAgent":
        """Compatibility loader that returns a stub instance."""
        agent = DummyMLAlphaZooAgent()
        agent.loaded = True
        return agent

    def predict(self, features):
        """
        Return neutral prediction (0) so that AlphaZoo does not
        influence FusionBrain unless properly enabled.
        """
        return 0.0

    def get_info(self):
        return {
            "type": "DummyMLAlphaZooAgent",
            "loaded": self.loaded,
        }
