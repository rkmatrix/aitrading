# ai/agents/offpolicy_learner.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = object
    optim = object

class MLP(nn.Module if torch else object):
    def __init__(self, obs_dim, act_dim):
        if torch is None:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        z = self.net(x)
        return self.mu(z), self.v(z)

class OffPolicyLearner:
    """
    Lightweight learner that can consume batches from ReplayBuffer.
    Supports a simple actor-critic update (AWAC-ish) for demonstration.
    """
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4, device: str = "cpu"):
        if torch is None:
            raise ImportError("Install torch to use OffPolicyLearner")
        self.device = device
        self.net = MLP(obs_dim, act_dim).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def update(self, batch: Tuple[np.ndarray, ...], isw: np.ndarray) -> Dict[str, float]:
        states, actions, rewards, next_states, dones, _infos = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(-1)
        isw = torch.tensor(isw.reshape(-1,1), dtype=torch.float32, device=self.device)

        mu, v = self.net(states)
        with torch.no_grad():
            _, v_next = self.net(next_states)
            target = rewards + (1 - dones) * 0.99 * v_next
        # Critic loss (MSE), weighted by IS weights
        critic_loss = ((v - target) ** 2 * isw).mean()

        # Simple actor regression toward actions (behavior cloning with value-weight)
        # Advantage proxy: (target - v).detach()
        adv = (target - v).detach()
        actor_loss = ((mu - actions) ** 2 * torch.relu(adv) * isw).mean()

        loss = actor_loss + critic_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "adv_mean": float(adv.mean().item()),
        }
