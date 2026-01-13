"""
ai/policy/agents/execution_ppo_agent.py

Compatibility stub for Phase 92 FusionBrain.

The real ExecutionPPOAgent is *not required* for Phase 92+ unless you
explicitly enable PPO-based decision overlays.

FusionBrain imports DummyExecutionPPOAgent as a fallback placeholder.
"""

from __future__ import annotations


class DummyExecutionPPOAgent:
    """
    A no-op PPO agent used only to maintain backward compatibility
    with older phases (Phases 68–71, 74, 77, 84).

    FusionBrain may call:
        - predict(obs)
        - load(path)
        - get_info()

    All are implemented as harmless stubs.
    """

    def __init__(self, *args, **kwargs):
        self.loaded = False

    @staticmethod
    def load(path: str) -> "DummyExecutionPPOAgent":
        """Compatibility loader – returns a fresh dummy agent."""
        agent = DummyExecutionPPOAgent()
        agent.loaded = True
        return agent

    def predict(self, obs):
        """
        Return a neutral (0) prediction.
        FusionBrain will ignore this if PPO is disabled in config.
        """
        return 0.0

    def get_info(self):
        """Return minimal metadata."""
        return {
            "type": "DummyExecutionPPOAgent",
            "loaded": self.loaded,
        }
