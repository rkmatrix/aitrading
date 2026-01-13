import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MetaPPOPolicy(nn.Module):
    """Actor-Critic policy with softmax head."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, temperature: float = 1.0):
        logits = self.actor(obs) / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()
        action = F.one_hot(idx, probs.shape[-1]).float()
        return action, dist.log_prob(idx), self.critic(obs)

    def evaluate(self, obs, action, temperature: float = 1.0):
        logits = self.actor(obs) / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        idx = action.argmax(-1)
        logp = dist.log_prob(idx)
        return logp, dist.entropy(), self.critic(obs)
