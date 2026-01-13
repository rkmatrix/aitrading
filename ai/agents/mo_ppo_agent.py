# ai/agents/mo_ppo_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

def _to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, discrete: bool, hidden=(256, 256)):
        super().__init__()
        self.discrete = discrete

        def mlp(in_dim, layers, out_dim):
            mods = []
            last = in_dim
            for h in layers:
                mods += [nn.Linear(last, h), nn.ReLU()]
                last = h
            mods += [nn.Linear(last, out_dim)]
            return nn.Sequential(*mods)

        self.actor_body = mlp(obs_dim, hidden, hidden[-1])
        if self.discrete:
            self.actor_head = nn.Linear(hidden[-1], act_dim)
        else:
            self.mu = nn.Linear(hidden[-1], act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.actor_body(obs)
        if self.discrete:
            logits = self.actor_head(z)
            return {"logits": logits, "value": self.critic(obs).squeeze(-1)}
        else:
            mu = self.mu(z)
            std = self.log_std.exp().clamp(min=1e-4, max=10.0)
            return {"mu": mu, "std": std, "value": self.critic(obs).squeeze(-1)}

    def act(self, obs: torch.Tensor):
        out = self.forward(obs)
        if self.discrete:
            dist = Categorical(logits=out["logits"])
            a = dist.sample()
            logp = dist.log_prob(a)
        else:
            dist = Normal(out["mu"], out["std"])
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
        v = out["value"]
        return a, logp, v

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs)
        if self.discrete:
            dist = Categorical(logits=out["logits"])
            return dist.log_prob(actions.long())
        else:
            dist = Normal(out["mu"], out["std"])
            return dist.log_prob(actions).sum(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)["value"]

class MOPPOAgent:
    """
    Multi-Objective PPO Agent with:
      - adaptive_update(...) for small online steps
      - optional EWC penalty to reduce catastrophic forgetting
      - ewc_estimate_fisher(...) to refresh consolidation anchors
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        discrete_action: bool,
        mo_weights: Optional[Dict[str, float]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        entropy_coef: float = 0.0,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        logger=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.logger = logger

        self.model = ActorCritic(obs_dim, act_dim, discrete=discrete_action).to(self.device)
        self.pi_opt = optim.Adam(self.model.parameters(), lr=policy_lr, eps=1e-5)
        self.v_opt = optim.Adam(self.model.critic.parameters(), lr=value_lr, eps=1e-5)

        self.mo_weights = mo_weights or {}

    def _maybe_log(self, **kw):
        if self.logger is None:
            return
        try:
            self.logger.log_adaptive(**kw)
        except Exception:
            pass

    def _compute_advantages(self, r, d, v, v2):
        delta = r + self.gamma * (1.0 - d) * v2 - v
        return delta

    @torch.no_grad()
    def ewc_estimate_fisher(self, batch: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Dict[str, Any]]]):
        """
        Estimate diagonal Fisher information using log-prob gradients on a mixed batch.
        Returns (params_snapshot, fisher_diag_dict)
        """
        if not batch:
            return None
        s = _to_t(np.stack([t[0] for t in batch]), self.device)
        a = _to_t(np.stack([t[1] for t in batch]), self.device)
        logp = self.model.log_prob(s, a)
        loss = -logp.mean()
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        fisher_diag = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            fisher_diag[name] = (p.grad.detach() ** 2).mean().clamp_(max=1e2)

        params_snapshot = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        return (params_snapshot, fisher_diag)

    def _ewc_penalty(self, ewc_state, strength: float) -> torch.Tensor:
        if not ewc_state or strength <= 0.0:
            return torch.tensor(0.0, device=self.device)
        params_snapshot, fisher_diag = ewc_state
        penalty = 0.0
        for name, p in self.model.named_parameters():
            if name in fisher_diag and name in params_snapshot:
                penalty += (fisher_diag[name] * (p - params_snapshot[name])**2).sum()
        return strength * penalty

    # --- Add inside your PPO agent class ---
    def push_experience(self, buffer, obs, action, next_obs, fill_dict, exposure: float = 0.0):
        from ai.execution.live_trade_sink import LiveTradeSink
        if not hasattr(self, "_phase27_sink"):
            # lazy-init with defaults if not supplied
            self._phase27_sink = LiveTradeSink(
                buffer=buffer,
                reward_conf={
                    "weights": {"pnl":1.0,"risk":-0.2,"cost":-0.05,"position":-0.01},
                    "pnl_norm_window": 2000, "dd_window": 1000, "clip": [-5.0, 5.0],
                },
            )
        return self._phase27_sink.on_step(obs, action, next_obs, fill_dict, exposure)

    def adaptive_update(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Dict[str, Any]]],
        epochs: int = 3,
        lr_scale: float = 1.0,
        kl_target: float = 0.02,
        max_grad_norm: float = 0.5,
        ewc_strength: float = 0.0,
        ewc_state=None,
    ) -> Dict[str, Any]:
        if not batch:
            return {}

        s = _to_t(np.stack([t[0] for t in batch]), self.device)
        a = _to_t(np.stack([t[1] for t in batch]), self.device)
        r = _to_t(np.array([t[2] for t in batch], dtype=np.float32), self.device)
        s2 = _to_t(np.stack([t[3] for t in batch]), self.device)
        d = _to_t(np.array([t[4] for t in batch], dtype=np.float32), self.device)

        with torch.no_grad():
            old_logp = self.model.log_prob(s, a)
            v = self.model.value(s)
            v2 = self.model.value(s2)
            adv = self._compute_advantages(r, d, v, v2)
            ret = adv + v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Temporarily scale LR (vol-aware factor passed in)
        for g in self.pi_opt.param_groups:
            g["lr"] = g["lr"] * float(lr_scale)
        for g in self.v_opt.param_groups:
            g["lr"] = g["lr"] * float(lr_scale)

        def _get_kl():
            with torch.no_grad():
                new_logp = self.model.log_prob(s, a)
            return (old_logp - new_logp).mean().abs().item()

        last_kl, pol_loss_out, val_loss_out, ewc_out = 0.0, 0.0, 0.0, 0.0

        for _ in range(epochs):
            logp = self.model.log_prob(s, a)
            ratio = torch.exp(logp - old_logp)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()

            out = self.model.forward(s)
            if self.model.discrete:
                entropy = Categorical(logits=out["logits"]).entropy().mean()
            else:
                entropy = Normal(out["mu"], out["std"]).entropy().sum(-1).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy

            value = out["value"]
            value_loss = nn.functional.huber_loss(value, ret, delta=1.0)

            penalty = self._ewc_penalty(ewc_state, ewc_strength)
            ewc_out = float(penalty.detach().cpu().item())

            self.pi_opt.zero_grad(set_to_none=True)
            (policy_loss + penalty).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.pi_opt.step()

            self.v_opt.zero_grad(set_to_none=True)
            (value_loss + 0.5 * penalty).backward()
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_grad_norm)
            self.v_opt.step()

            last_kl = _get_kl()
            if last_kl > 1.5 * kl_target:
                break

            pol_loss_out = policy_loss.item()
            val_loss_out = value_loss.item()

        # Gentle auto-tune of clip ratio based on KL
        if last_kl < 0.5 * kl_target:
            self.clip_ratio = float(np.clip(self.clip_ratio * 1.02, 0.1, 0.4))
        elif last_kl > 1.5 * kl_target:
            self.clip_ratio = float(np.clip(self.clip_ratio * 0.98, 0.05, 0.3))

        out_metrics = {
            "policy_loss": float(pol_loss_out),
            "value_loss": float(val_loss_out),
            "kl": float(last_kl),
            "ewc_penalty": float(ewc_out),
            "clip_ratio": float(self.clip_ratio),
            "batch": len(batch),
            "epochs": epochs,
        }
        self._maybe_log(kind="adaptive_update", **out_metrics)
        return out_metrics
