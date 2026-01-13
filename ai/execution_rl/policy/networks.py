from __future__ import annotations
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

class SmallMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)
