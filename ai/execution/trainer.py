# -*- coding: utf-8 -*-
from dataclasses import dataclass
import os
import numpy as np
from .executor_env import ExecutionEnv
from .rl_agent import ExecutionRLAgent

@dataclass
class TrainConfig:
    episodes: int = 50
    max_steps: int = 60
    save_every: int = 10
    out_dir: str = "models/phase20"
    seed: int = 42

class ExecutionTrainer:
    def __init__(self, env: ExecutionEnv, agent: ExecutionRLAgent, cfg: TrainConfig):
        self.env = env
        self.agent = agent
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        np.random.seed(cfg.seed)

    def run(self):
        rewards = []
        for ep in range(1, self.cfg.episodes + 1):
            state = self.env.reset()
            ep_reward = 0.0
            for _ in range(self.cfg.max_steps):
                a = self.agent.act(state, explore=True)
                ns, r, done, info = self.env.step(a)
                self.agent.push(state, a, r, ns, done)
                state = ns
                ep_reward += r
                self.agent.learn()
                if done:
                    break
            rewards.append(ep_reward)
            if ep % self.cfg.save_every == 0:
                path = os.path.join(self.cfg.out_dir, f"exec_rl_ep{ep}.pt")
                self.agent.save(path)
        self.agent.save(os.path.join(self.cfg.out_dir, "exec_rl_final.pt"))
        return rewards
