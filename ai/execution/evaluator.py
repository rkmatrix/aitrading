# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
from .executor_env import ExecutionEnv
from .rl_agent import ExecutionRLAgent

@dataclass
class EvalResult:
    avg_reward: float
    avg_slippage_bps: float

class ExecutionEvaluator:
    def __init__(self, env: ExecutionEnv, agent: ExecutionRLAgent):
        self.env = env
        self.agent = agent

    def evaluate(self, episodes: int = 10) -> EvalResult:
        rewards, slips = [], []
        for _ in range(episodes):
            s = self.env.reset()
            ep_r, ep_slip = 0.0, []
            while True:
                a = self.agent.act(s, explore=False)
                ns, r, done, info = self.env.step(a)
                s = ns
                ep_r += r
                ep_slip.append(info.get("slippage_bps", 0.0))
                if done:
                    break
            rewards.append(ep_r)
            slips.append(np.mean(ep_slip) if ep_slip else 0.0)
        return EvalResult(avg_reward=float(np.mean(rewards)), avg_slippage_bps=float(np.mean(slips)))
