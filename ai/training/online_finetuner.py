# ai/training/online_finetuner.py
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Deque

import numpy as np
import torch
from stable_baselines3 import PPO

log = logging.getLogger("OnlineFineTuner")


@dataclass
class OnlineFineTuneConfig:
    policy_path: Path
    save_path: Path
    regime_state_file: Path

    max_buffer_size: int
    batch_size: int
    mini_updates: int
    update_interval_steps: int
    min_buffer_for_update: int
    lr_scale_risk_off: float
    skip_in_risk_off: bool


class OnlineFineTuner:
    def __init__(self, cfg: Dict[str, Any]):
        paths = cfg.get("paths", {})
        ot = cfg.get("online_finetune", {})

        self.cfg = OnlineFineTuneConfig(
            policy_path=Path(paths["policy_path"]),
            save_path=Path(paths["save_path"]),
            regime_state_file=Path(paths["regime_state_file"]),
            max_buffer_size=int(ot["max_buffer_size"]),
            batch_size=int(ot["batch_size"]),
            mini_updates=int(ot["mini_updates"]),
            update_interval_steps=int(ot["update_interval_steps"]),
            min_buffer_for_update=int(ot["min_buffer_for_update"]),
            lr_scale_risk_off=float(ot["lr_scale_risk_off"]),
            skip_in_risk_off=bool(ot["skip_in_risk_off"]),
        )

        self.model: PPO = PPO.load(self.cfg.policy_path, print_system_info=False)
        self.policy = self.model.policy
        self.optimizer = self.policy.optimizer

        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.max_buffer_size)
        self.step_count = 0
        self.update_count = 0

    def observe_transition(self, s, a, r, ns, done, info=None):
        self.buffer.append(
            {"state": s, "action": a, "reward": r, "next": ns, "done": done}
        )
        self.step_count += 1

        if self.step_count % self.cfg.update_interval_steps == 0:
            self._maybe_update()

    def _regime(self) -> str:
        fp = self.cfg.regime_state_file
        if not fp.exists():
            return "neutral"
        try:
            data = json.loads(fp.read_text())
            return data.get("regime", "neutral")
        except Exception:
            return "neutral"

    def _maybe_update(self):
        if len(self.buffer) < self.cfg.min_buffer_for_update:
            return

        regime = self._regime()

        if regime == "risk_off" and self.cfg.skip_in_risk_off:
            return

        lr_scale = 1.0
        if regime == "risk_off":
            lr_scale = self.cfg.lr_scale_risk_off

        # adjust lr
        base_lr = []
        for g in self.optimizer.param_groups:
            base_lr.append(g["lr"])
            g["lr"] = g["lr"] * lr_scale

        for _ in range(self.cfg.mini_updates):
            if len(self.buffer) < self.cfg.batch_size:
                break

            batch = np.random.choice(len(self.buffer), self.cfg.batch_size, replace=False)
            states = np.stack([self.buffer[i]["state"] for i in batch])
            rewards = np.asarray([self.buffer[i]["reward"] for i in batch])

            s = torch.as_tensor(states, dtype=torch.float32)
            r = torch.as_tensor(rewards, dtype=torch.float32)

            v = self.policy.predict_values(s)
            loss = ((v.flatten() - r) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        # restore lr
        for g, lr in zip(self.optimizer.param_groups, base_lr):
            g["lr"] = lr

        self.update_count += 1
        self._save()

    def _save(self):
        self.model.save(self.cfg.save_path)
