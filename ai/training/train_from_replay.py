# ai/training/train_from_replay.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO

log = logging.getLogger("TrainFromReplay")


# ======================================================================
# Replay → Dataset
# ======================================================================

@dataclass
class ReplaySample:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.items = self._convert(rows)

    def _make_state(self, row: Dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                float(row.get("price", 0.0)),
                float(row.get("position", 0.0)),
                float(row.get("equity", 0.0)),
                float(row.get("risk", 0.0)),
            ],
            dtype=np.float32,
        )

    def _convert(self, rows: List[Dict[str, Any]]) -> List[ReplaySample]:
        samples: List[ReplaySample] = []

        if len(rows) >= 2:
            for i in range(len(rows) - 1):
                cur, nxt = rows[i], rows[i + 1]
                samples.append(
                    ReplaySample(
                        state=self._make_state(cur),
                        action=float(cur.get("position", 0.0)),
                        reward=float(cur.get("reward", 0.0)),
                        next_state=self._make_state(nxt),
                        done=False,
                    )
                )
            return samples

        if len(rows) == 1:
            cur = rows[0]
            s = self._make_state(cur)
            samples.append(
                ReplaySample(
                    state=s,
                    action=float(cur.get("position", 0.0)),
                    reward=float(cur.get("reward", 0.0)),
                    next_state=s.copy(),
                    done=False,
                )
            )
            return samples

        raise ValueError("Replay dataset is empty — cannot train.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        return s.state, s.action, s.reward, s.next_state, s.done


# ======================================================================
# Trainer
# ======================================================================

@dataclass
class TrainConfig:
    policy_path: Path
    replay_path: Path
    lr: float = 3e-4
    batch_size: int = 64
    gradient_steps: int = 2000
    save_to: Path = Path("models/policies/EquityRLPolicy/trained")


class ReplayTrainer:
    """
    Offline Replay Trainer (PPO-based) used by Phase 56.
    """

    def __init__(self, cfg: Union[TrainConfig, Dict[str, Any]], replay_rows: List[Dict[str, Any]]):
        # Normalize cfg dict → dataclass
        if isinstance(cfg, TrainConfig):
            self.cfg = cfg
        else:
            self.cfg = TrainConfig(
                policy_path=Path(cfg.get("policy_path", "models/policies/EquityRLPolicy/model.zip")),
                replay_path=Path(
                    cfg.get(
                        "replay_path",
                        cfg.get("paths", {}).get("replay_file", "data/replay/phase55_replay.jsonl"),
                    )
                ),
                lr=float(cfg.get("lr", 3e-4)),
                batch_size=int(cfg.get("batch_size", 64)),
                gradient_steps=int(cfg.get("gradient_steps", 2000)),
                save_to=Path(
                    cfg.get(
                        "save_to",
                        cfg.get("paths", {}).get("policies_root", "models/policies/EquityRLPolicy/model"),
                    )
                ),
            )

        self.replay_rows = replay_rows
        self.stats: Dict[str, float] = {}
        self.num_episodes: int = 0

    # -------------------------
    # NO NORMALIZATION: SB3 needs .zip
    # -------------------------
    def normalize_policy_path(self, p: Path) -> Path:
        return p  # keep .zip

    # -------------------------
    # Load PPO model
    # -------------------------
    def load_model(self):
        safe_path = self.normalize_policy_path(self.cfg.policy_path)
        log.info("Loading PPO policy from %s", safe_path)
        return PPO.load(safe_path, print_system_info=False)

    # -------------------------
    # Training loop
    # -------------------------
    def train(self):
        import torch  # needed for backward

        rows = self.replay_rows
        if not rows:
            raise ValueError("ReplayTrainer received empty replay_rows")

        dataset = ReplayDataset(rows)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        model = self.load_model()
        policy = model.policy
        optimizer = policy.optimizer

        log.info(
            "Starting offline RL training for %d gradient steps (batch_size=%d)",
            self.cfg.gradient_steps,
            self.cfg.batch_size,
        )

        steps = 0
        rewards_collected: List[float] = []

        while steps < self.cfg.gradient_steps:
            for batch in loader:
                states, actions, rewards, next_states, dones = batch

                states = states.float()
                rewards = rewards.float()

                rewards_collected.extend(rewards.cpu().numpy().tolist())

                values = policy.predict_values(states)
                loss = ((values.flatten() - rewards) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                if steps >= self.cfg.gradient_steps:
                    break

        # Save PPO model
        self.cfg.save_to.mkdir(parents=True, exist_ok=True)
        out_path = self.cfg.save_to / "model.zip"
        log.info("Saving trained model → %s", out_path)
        model.save(out_path)

        # Metrics for Phase57
        r = np.array(rewards_collected) if rewards_collected else np.array([0.0])

        self.stats = {
            "avg_reward": float(np.mean(r)),
            "sharpe": float(np.mean(r) / (np.std(r) + 1e-6)),
            "winrate": float(np.sum(r > 0) / len(r)),
            "max_drawdown": float(np.min(r)),
        }

        self.num_episodes = len(rows)
        log.info("📈 Training Stats: %s", self.stats)

        return out_path

    # -------------------------
    # Required by bundle_writer
    # -------------------------
    def get_model_weights(self):
        # Placeholder; bundle_writer will prefer copying model.zip
        return np.array([1.0, 2.0, 3.0])
