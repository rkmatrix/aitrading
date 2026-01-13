"""
MetaController
--------------
Phase-14/15 adaptive meta-policy that tunes allocator or learner hyperparameters
based on recent performance metrics aggregated by MetaFeedbackIngestor.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MetaController:
    """
    Maintains meta-level parameters controlling the allocator or learner.
    Can ingest feedback payloads to self-adjust learning rates, risk weights,
    or exploration factors.
    """

    def __init__(self,
                 lr: float = 1e-3,
                 risk_scale: float = 1.0,
                 exploration_temp: float = 1.0,
                 save_dir: str = "data"):
        self.lr = lr
        self.risk_scale = risk_scale
        self.exploration_temp = exploration_temp
        self.last_update = None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.save_dir / "meta_state.json"

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------
    def update_from_feedback(self, payload: Dict[str, float]) -> None:
        """
        Adjust meta parameters based on incoming feedback payload.

        payload fields:
            avg_reward   : float
            success_rate : float
        """
        if not payload:
            return

        avg_reward = payload.get("avg_reward", 0.0)
        success = payload.get("success_rate", 0.5)

        # Simple proportional adjustments (can be replaced with PID logic)
        self.lr *= (1.0 + 0.1 * (avg_reward))
        self.risk_scale *= (1.0 + 0.2 * (success - 0.5))
        self.exploration_temp *= max(0.9, 1.0 - 0.1 * success)

        # Clamp to safe bounds
        self.lr = float(max(min(self.lr, 1e-2), 1e-5))
        self.risk_scale = float(max(min(self.risk_scale, 2.0), 0.5))
        self.exploration_temp = float(max(min(self.exploration_temp, 2.0), 0.2))

        self.last_update = datetime.utcnow().isoformat()

        # Persist changes
        self.save_meta_state()
        print(f"✅ MetaController updated: lr={self.lr:.5f}, "
              f"risk_scale={self.risk_scale:.3f}, "
              f"exploration_temp={self.exploration_temp:.3f}")

    # ------------------------------------------------------------------
    def save_meta_state(self, path: str = None) -> None:
        """Save current meta-state to JSON."""
        out_path = Path(path) if path else self.state_path
        state = {
            "lr": self.lr,
            "risk_scale": self.risk_scale,
            "exploration_temp": self.exploration_temp,
            "last_update": self.last_update,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    def load_meta_state(self, path: str = None) -> None:
        """Load meta-state from JSON if present."""
        in_path = Path(path) if path else self.state_path
        if not in_path.exists():
            return
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.lr = state.get("lr", self.lr)
            self.risk_scale = state.get("risk_scale", self.risk_scale)
            self.exploration_temp = state.get("exploration_temp", self.exploration_temp)
            self.last_update = state.get("last_update", self.last_update)
            print(f"📂 Loaded meta-state from {in_path}")
        except Exception as e:
            print(f"⚠️  Failed to load meta-state: {e}")

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""
        return {
            "lr": self.lr,
            "risk_scale": self.risk_scale,
            "exploration_temp": self.exploration_temp,
            "last_update": self.last_update,
        }

    def __repr__(self):
        return (f"MetaController(lr={self.lr:.5f}, "
                f"risk_scale={self.risk_scale:.3f}, "
                f"exploration_temp={self.exploration_temp:.3f})")
