# ai/policy/alpha_brain_fused_loss.py
"""
AlphaBrain Fused Loss Helpers

This module provides utilities to add an auxiliary "fused target" loss
on top of standard PPO training, *without* modifying stable-baselines3
internal code.

Usage pattern (conceptual):

    from stable_baselines3 import PPO
    from ai.policy.alpha_brain_fused_loss import fused_aux_loss_step

    model = PPO.load("models/policies/AlphaBrainPolicy/model.zip", device="cpu")

    # During a custom training loop, after a PPO mini-update:
    fused_aux_loss_step(
        model=model,
        obs_batch=obs_np,            # shape [N, obs_dim]
        fused_targets=fused_np,      # shape [N, action_dim or 1]
        lambda_aux=0.05,
    )

Then save the model as usual.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO


def fused_aux_loss_step(
    *,
    model: PPO,
    obs_batch: np.ndarray,
    fused_targets: np.ndarray,
    lambda_aux: float = 0.05,
) -> Optional[float]:
    """
    Applies a small auxiliary MSE loss step between the policy output and a fused target.

    Args:
        model: PPO instance (SB3).
        obs_batch: observations, shape [N, obs_dim]
        fused_targets: target signal, shape [N, A] or [N, 1]
        lambda_aux: weight of auxiliary loss.

    Returns:
        loss_value (float) if training step performed, else None.
    """
    if lambda_aux <= 0.0:
        return None

    device = model.device
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(fused_targets, dtype=torch.float32, device=device)

    # Forward pass through policy's actor net
    with torch.no_grad():
        features = model.policy.features_extractor(obs_tensor)
    latent_pi = model.policy.mlp_extractor.policy_net(features)
    logits = model.policy.action_net(latent_pi)

    # For continuous actions (Box), SB3 uses a Gaussian; treat logits as mean
    # Adjust if your action space is discrete.
    if target_tensor.shape != logits.shape:
        # Simple broadcast if possible
        target_tensor = target_tensor.view_as(logits)

    mse = torch.nn.functional.mse_loss(logits, target_tensor)

    loss = lambda_aux * mse

    model.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.policy.max_grad_norm)
    model.policy.optimizer.step()

    return float(loss.item())
