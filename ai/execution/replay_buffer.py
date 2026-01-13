# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

@dataclass
class ReplayBuffer:
    capacity: int
    _cursor: int = 0
    _size: int = 0

    def __post_init__(self):
        self.states = None
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.empty(self.capacity, dtype=np.bool_)

    def push(self, transition: Transition):
        s = transition.state.astype(np.float32)
        ns = transition.next_state.astype(np.float32)
        if self.states is None:
            self.states = np.empty((self.capacity, s.shape[0]), dtype=np.float32)
            self.next_states = np.empty((self.capacity, ns.shape[0]), dtype=np.float32)
        self.states[self._cursor] = s
        self.actions[self._cursor] = transition.action
        self.rewards[self._cursor] = transition.reward
        self.next_states[self._cursor] = ns
        self.dones[self._cursor] = transition.done
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self._size
