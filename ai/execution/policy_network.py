# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class NetConfig:
    state_dim: int
    action_dim: int
    hidden: Tuple[int, int] = (128, 128)
    dropout: float = 0.1

class MLP(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        h1, h2 = cfg.hidden
        self.fc1 = nn.Linear(cfg.state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h2, cfg.action_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.head(x)  # Q-values (DQN-style)

class ExecutionPolicy(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.net = MLP(cfg)

    def forward(self, state):
        return self.net(state)
