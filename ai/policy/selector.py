# ai/policy/selector.py
from __future__ import annotations
import math, json, random, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

class SelectorState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = {"trials": {}, "rewards": {}, "last_pick_ts": 0.0}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def update(self, pid: str, reward: float, decay: float = 1.0):
        s = self.state
        s["trials"].setdefault(pid, 0)
        s["rewards"].setdefault(pid, 0.0)
        # decay existing rewards to prefer recent performance
        if 0 < decay < 1.0:
            for k in list(s["rewards"].keys()):
                s["rewards"][k] *= decay
        s["trials"][pid] += 1
        s["rewards"][pid] += float(reward)
        s["last_pick_ts"] = time.time()

    def arms(self) -> List[str]:
        return list(self.state.get("trials", {}).keys())

    def ensure_arms(self, pids: List[str]):
        for p in pids:
            self.state["trials"].setdefault(p, 0)
            self.state["rewards"].setdefault(p, 0.0)

def ucb1_pick(pids: List[str], trials: Dict[str, int], rewards: Dict[str, float], min_trials: int) -> str:
    total = sum(trials.values()) + 1e-9
    for p in pids:
        if trials.get(p, 0) < min_trials:
            return p
    # compute UCB score
    def mean(p): 
        t = max(trials.get(p,0), 1)
        return rewards.get(p,0.0) / t
    best_p, best_score = None, -1e18
    for p in pids:
        t = max(trials.get(p,0), 1)
        bonus = math.sqrt(2.0 * math.log(total) / t)
        score = mean(p) + bonus
        if score > best_score:
            best_p, best_score = p, score
    return best_p

def epsilon_greedy_pick(pids: List[str], trials: Dict[str,int], rewards: Dict[str,float], eps: float, min_trials: int) -> str:
    for p in pids:
        if trials.get(p,0) < min_trials:
            return p
    if random.random() < eps:
        return random.choice(pids)
    # exploit
    def mean(p):
        t = max(trials.get(p,0), 1)
        return rewards.get(p,0.0)/t
    return max(pids, key=mean)

def softmax_pick(pids: List[str], trials: Dict[str,int], rewards: Dict[str,float], temperature: float, min_trials: int) -> str:
    for p in pids:
        if trials.get(p,0) < min_trials:
            return p
    def mean(p):
        t = max(trials.get(p,0), 1)
        return rewards.get(p,0.0)/t
    vals = [mean(p) for p in pids]
    mx = max(vals) if vals else 0.0
    exps = [math.exp((v - mx)/max(temperature,1e-6)) for v in vals]
    s = sum(exps) + 1e-12
    r = random.random()*s
    c = 0.0
    for p, e in zip(pids, exps):
        c += e
        if r <= c:
            return p
    return pids[-1]

class PolicySelector:
    def __init__(self, state_path: Path, strategy: str = "ucb1", epsilon: float = 0.1, temperature: float = 0.7, min_trials: int = 1, decay: float = 1.0):
        self.state = SelectorState(state_path)
        self.strategy = strategy
        self.epsilon = epsilon
        self.temperature = temperature
        self.min_trials = min_trials
        self.decay = decay

    def pick(self, candidates: List[str]) -> str:
        self.state.ensure_arms(candidates)
        s = self.state.state
        trials, rewards = s["trials"], s["rewards"]
        if self.strategy == "ucb1":
            pid = ucb1_pick(candidates, trials, rewards, self.min_trials)
        elif self.strategy == "epsilon_greedy":
            pid = epsilon_greedy_pick(candidates, trials, rewards, self.epsilon, self.min_trials)
        elif self.strategy == "softmax":
            pid = softmax_pick(candidates, trials, rewards, self.temperature, self.min_trials)
        else:
            pid = random.choice(candidates)
        s["last_pick_ts"] = time.time()
        self.state.save()
        return pid

    def record_reward(self, pid: str, reward: float):
        self.state.update(pid, reward, decay=self.decay)
        self.state.save()
