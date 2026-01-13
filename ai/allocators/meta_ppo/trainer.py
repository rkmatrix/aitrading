import numpy as np, torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict
from .buffer import OnPolicyBuffer, Transition
from .policy import MetaPPOPolicy
from .reward import RewardEngine, RewardConfig
from .meta_controller import MetaController
from .telemetry import Telemetry


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.15
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    minibatch_size: int = 256
    update_steps: int = 2


class OnlineAllocatorTrainer:
    def __init__(self, obs_dim, act_dim, ppo_cfg: PPOConfig,
                 reward_cfg: RewardConfig, meta: MetaController, telemetry: Telemetry, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = MetaPPOPolicy(obs_dim, act_dim).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=ppo_cfg.lr)
        self.ppo_cfg = ppo_cfg
        self.reward_engine = RewardEngine(reward_cfg)
        self.meta = meta
        self.telemetry = telemetry

    def update_meta(self, regime: Dict[str, float]):
        p = self.meta.select(regime)
        for g in self.optim.param_groups:
            g["lr"] = p.lr
        self.ppo_cfg.entropy_coef = p.entropy_coef
        self.reward_engine.cfg.lambda_drawdown = p.lambda_drawdown
        self.reward_engine.cfg.kappa_cost = p.kappa_cost
        return p

    def ppo_update(self, buf: OnPolicyBuffer, temperature: float) -> Dict[str, float]:
        obs, act, logp_old, rew, val, done = buf.get()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act, dtype=torch.float32, device=self.device)
        logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=self.device)
        val_t = torch.tensor(val, dtype=torch.float32, device=self.device)
        adv = torch.tensor(rew, dtype=torch.float32, device=self.device) - val_t

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.tensor(rew, dtype=torch.float32, device=self.device)
        n, mb = obs_t.size(0), self.ppo_cfg.minibatch_size
        losses = {"pi": [], "vf": [], "ent": []}
        for _ in range(self.ppo_cfg.update_steps):
            for s in range(0, n, mb):
                j = torch.arange(s, min(s + mb, n))
                logp, ent, v = self.policy.evaluate(obs_t[j], act_t[j], temperature)
                ratio = torch.exp(logp - logp_old_t[j])
                surr1, surr2 = ratio * adv[j], torch.clamp(ratio, 1 - self.ppo_cfg.clip_eps, 1 + self.ppo_cfg.clip_eps) * adv[j]
                pi_loss = -torch.min(surr1, surr2).mean() - self.ppo_cfg.entropy_coef * ent.mean()
                vf_loss = F.mse_loss(v, ret[j]) * self.ppo_cfg.vf_coef
                loss = pi_loss + vf_loss
                self.optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optim.step()
                losses["pi"].append(pi_loss.item()); losses["vf"].append(vf_loss.item()); losses["ent"].append(ent.mean().item())
        return {k+"_loss": float(np.mean(v)) for k, v in losses.items()}

    def compute_reward(self, pnl, costs, drawdown, turnover):
        return self.reward_engine.compute(pnl, costs, drawdown, turnover)
