# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .policy_network import ExecutionPolicy, NetConfig
from .replay_buffer import ReplayBuffer, Transition

@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    start_random: int = 512
    eps_init: float = 1.0
    eps_final: float = 0.05
    eps_decay_steps: int = 20_000
    target_tau: float = 0.005
    device: str = "cpu"

class ExecutionRLAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: AgentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        net_cfg = NetConfig(state_dim=state_dim, action_dim=action_dim)
        self.q = ExecutionPolicy(net_cfg).to(self.device)
        self.q_target = ExecutionPolicy(net_cfg).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self._steps = 0

    def epsilon(self) -> float:
        d = min(self._steps, self.cfg.eps_decay_steps)
        return self.cfg.eps_final + (self.cfg.eps_init - self.cfg.eps_final) * (1 - d / self.cfg.eps_decay_steps)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        self._steps += 1
        if explore and (len(self.buffer) < self.cfg.start_random or np.random.rand() < self.epsilon()):
            return np.random.randint(0, self.q.net.head.out_features)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qv = self.q(st)
            return int(torch.argmax(qv, dim=1).item())

    def push(self, *args):
        self.buffer.push(Transition(*args))

    def learn(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=self.device)

        q = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_max = self.q_target(ns).max(1).values
            target = r + (1 - d) * self.cfg.gamma * next_max
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)
        return float(loss.item()), float(q.mean().item())

    def save(self, path: str):
        torch.save({"model": self.q.state_dict(), "steps": self._steps}, path)

    def load(self, path: str, map_location: str = "cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.q.load_state_dict(ckpt["model"])
        self.q_target.load_state_dict(self.q.state_dict())
        self._steps = int(ckpt.get("steps", 0))
