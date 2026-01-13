# ai/execution/online_finetune_bridge.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from ai.training.online_finetuner import OnlineFineTuner

log = logging.getLogger("OnlineFineTuneBridge")


class OnlineFineTuneBridge:
    def __init__(self, cfg_path: str = "configs/phase60_online_finetune.yaml"):
        self.cfg_path = Path(cfg_path)
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Online fine-tune config not found: {self.cfg_path}")

        with self.cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.tuner = OnlineFineTuner(cfg)
        log.info("OnlineFineTuneBridge initialized with %s", self.cfg_path)

    def encode(self, x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=np.float32)
        try:
            return np.asarray([float(x)], dtype=np.float32)
        except Exception:
            return np.zeros(1, dtype=np.float32)

    def observe(self, obs, action, reward, next_obs, done: bool, info: Dict[str, Any] | None = None):
        s = self.encode(obs)
        ns = self.encode(next_obs)
        self.tuner.observe_transition(s, float(action), float(reward), ns, bool(done), info)
