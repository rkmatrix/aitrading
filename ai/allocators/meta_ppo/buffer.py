import numpy as np
from dataclasses import dataclass


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    logp: float
    reward: float
    value: float
    done: bool


class OnPolicyBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.logp = np.zeros(capacity, np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.val = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.full = False

    def add(self, t: Transition):
        i = self.ptr
        self.obs[i] = t.obs
        self.act[i] = t.action
        self.logp[i] = t.logp
        self.rew[i] = t.reward
        self.val[i] = t.value
        self.done[i] = float(t.done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.full |= self.ptr == 0

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def get(self):
        n = self.size()
        return self.obs[:n], self.act[:n], self.logp[:n], self.rew[:n], self.val[:n], self.done[:n]

    def clear(self):
        self.ptr = 0
        self.full = False
