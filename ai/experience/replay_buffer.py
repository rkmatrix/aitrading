# ai/experience/replay_buffer.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

@dataclass
class Transition:
    # Generic RL tuple
    state: np.ndarray          # shape = (obs_dim,)
    action: np.ndarray         # shape = (act_dim,) or scalar
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict                # any extra (order_id, symbol, ts, costs, slippage, etc.)

class SumTree:
    """Efficient sampling by priority (for PER)."""
    def __init__(self, capacity: int):
        # next power of two for simplicity
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity <<= 1
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.data: List[Optional[Transition]] = [None] * self.capacity
        self.write_ptr = 0
        self.size = 0

    def _update(self, idx: int, value: float):
        change = value - self.tree[idx]
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def total(self) -> float:
        return self.tree[1]

    def add(self, p: float, data: Transition):
        leaf = self.write_ptr + self.capacity
        self.data[self.write_ptr] = data
        self._update(leaf, p)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, Transition]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]

class ReplayBuffer:
    """Uniform or PER replay buffer with numpy arrays. Designed for continuous updates from live trades."""
    def __init__(
        self,
        capacity: int = 250_000,
        prioritized: bool = True,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_eps = per_eps
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0

        self.states = None
        self.actions = None
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.infos: List[Optional[dict]] = [None] * capacity

        self.tree = SumTree(capacity) if prioritized else None
        self.max_priority = 1.0

        self._obs_dim = None
        self._act_dim = None

    def _init_arrays(self, state: np.ndarray, action: np.ndarray):
        if self.states is None:
            self._obs_dim = int(np.prod(state.shape))
            self.states = np.zeros((self.capacity, self._obs_dim), dtype=np.float32)
        if self.actions is None:
            self._act_dim = int(np.prod(action.shape)) if hasattr(action, "shape") else 1
            self.actions = np.zeros((self.capacity, self._act_dim), dtype=np.float32)
        if self.next_states is None:
            self.next_states = np.zeros((self.capacity, self._obs_dim), dtype=np.float32)

    def push(self, tr: Transition, priority: Optional[float] = None):
        self._init_arrays(tr.state, tr.action)

        i = self.ptr
        self.states[i] = tr.state.reshape(-1)
        self.actions[i] = np.array(tr.action).reshape(-1)
        self.rewards[i] = float(tr.reward)
        self.next_states[i] = tr.next_state.reshape(-1)
        self.dones[i] = bool(tr.done)
        self.infos[i] = tr.info or {}
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if self.prioritized:
            p = (abs(priority if priority is not None else tr.reward) + self.per_eps) ** self.per_alpha
            self.tree.add(p, tr)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int):
        assert self.size > 0, "Empty buffer"
        idxs = None
        weights = None

        if self.prioritized:
            segment = self.tree.total() / batch_size
            idxs = []
            samples = []
            ps = []
            for i in range(batch_size):
                s = random.random() * segment + i * segment
                data_idx, p, tr = self.tree.get(s)
                # If item got overwritten since, skip silently
                if tr is None:
                    continue
                idxs.append(data_idx)
                samples.append(tr)
                ps.append(p)
            ps = np.array(ps) + self.per_eps
            probs = ps / (ps.sum() + 1e-12)
            weights = (len(self) * probs) ** (-self.per_beta)
            weights /= weights.max() + 1e-12

            # Convert list[Transition] -> arrays
            states = np.stack([t.state.reshape(-1) for t in samples], axis=0)
            actions = np.stack([np.array(t.action).reshape(-1) for t in samples], axis=0)
            rewards = np.array([t.reward for t in samples], dtype=np.float32)
            next_states = np.stack([t.next_state.reshape(-1) for t in samples], axis=0)
            dones = np.array([t.done for t in samples], dtype=np.bool_)
            infos = [t.info for t in samples]
            return idxs, weights.astype(np.float32), (states, actions, rewards, next_states, dones, infos)

        # Uniform sampling
        idxs = self.rng.integers(low=0, high=self.size, size=batch_size)
        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]
        infos = [self.infos[i] for i in idxs]
        weights = np.ones(len(idxs), dtype=np.float32)
        return idxs.tolist(), weights, (states, actions, rewards, next_states, dones, infos)

    def update_priorities(self, idxs: List[int], new_priorities: np.ndarray):
        if not self.prioritized:
            return
        for i, p in zip(idxs, new_priorities):
            leaf = i + self.tree.capacity
            pr = (abs(float(p)) + self.per_eps) ** self.per_alpha
            # Update tree along path
            self.tree._update(leaf, pr)
            self.max_priority = max(self.max_priority, pr)
