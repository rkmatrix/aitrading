import numpy as np, random, collections

class PrioritizedReplay:
    def __init__(self, capacity=200000, alpha=0.6, beta0=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta0 = beta0
        self.mem = []
        self.priorities = []
        self.pos = 0
        self.t = 0  # steps seen

    def add(self, transition, td_error=1.0):
        p = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.mem) < self.capacity:
            self.mem.append(transition)
            self.priorities.append(p)
        else:
            self.mem[self.pos] = transition
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity
        self.t += 1

    def sample(self, batch_size):
        if len(self.mem) == 0:
            return [], [], 0.0
        probs = np.array(self.priorities, dtype=float)
        probs /= probs.sum()
        idx = np.random.choice(len(self.mem), batch_size, p=probs)
        samples = [self.mem[i] for i in idx]
        beta = min(1.0, self.beta0 + 0.00001 * self.t)
        weights = (len(self.mem) * probs[idx]) ** (-beta)
        weights /= weights.max()
        return samples, weights, idx

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = (abs(float(e)) + 1e-6) ** self.alpha

    def __len__(self): return len(self.mem)
