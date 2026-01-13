# ai/ensemble/meta_controller.py
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = object


class SimpleMetaNet(nn.Module if torch else object):
    """Small MLP that converts context → blend weights via softmax."""
    def __init__(self, input_dim: int, num_policies: int, hidden: int = 32):
        if torch:
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, num_policies)
        else:
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(size=(input_dim, hidden)) * 0.1
            self.b1 = np.zeros((hidden,))
            self.W2 = rng.normal(size=(hidden, hidden)) * 0.1
            self.b2 = np.zeros((hidden,))
            self.Wo = rng.normal(size=(hidden, num_policies)) * 0.1
            self.bo = np.zeros((num_policies,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if torch:
            import torch.nn.functional as F
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            h = torch.tanh(self.l1(xt))
            h = torch.tanh(self.l2(h))
            logits = self.out(h)
            w = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            return w
        else:
            h = np.tanh(x @ self.W1 + self.b1)
            h = np.tanh(h @ self.W2 + self.b2)
            logits = h @ self.Wo + self.bo
            exps = np.exp(logits - np.max(logits))
            return exps / (exps.sum() + 1e-12)
