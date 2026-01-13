import numpy as np
from .base import ExecutionPolicy

class LinUCB(ExecutionPolicy):
    """Choose aggression level / size bucket with contextual bandit."""
    def __init__(self, n_actions:int=5, d:int=6, alpha:float=0.5):
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros((d, 1)) for _ in range(n_actions)]
        self.n_actions = n_actions

    def _phi(self, s):
        return np.array([
            s.get("spread",0), s.get("imbalance",0), s.get("volatility",0),
            s.get("remaining_time",1), s.get("remaining_qty",1), s.get("micro_alpha",0)
        ], dtype=float).reshape(-1,1)

    def act(self, state):
        x = self._phi(state)
        p = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float((theta.T @ x)[0,0])
            conf = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            p.append(mean + conf)
        a_star = int(np.argmax(p))
        # Convert action to (size, aggression) discretization
        size = state.get("remaining_qty", 1) // max(self.n_actions,1)
        agg = ["passive","mid","market","mid","passive"][a_star % 5]
        return {"size": max(size,1), "aggression": agg, "price": None}

    def update(self, state, action_idx:int, reward:float):
        x = self._phi(state)
        self.A[action_idx] += x @ x.T
        self.b[action_idx] += reward * x
