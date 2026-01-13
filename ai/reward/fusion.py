import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

def _safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0

def _safe_std(x):
    return float(np.std(x)) if len(x) else 0.0

class RewardFusion:
    """
    Phase 28+ adaptive reward fusion engine.
    Adds momentum bias, Sharpe-style boost, and RL feedback weighting
    (e.g., via policy confidence or advantage) to the fused reward.
    """

    def __init__(self, cfg):
        fusion_cfg = cfg.get("reward_fusion", {})
        self.window = int(fusion_cfg.get("window", 100))

        # Base weights
        w = fusion_cfg.get("weights", {})
        self.w_pnl       = float(w.get("pnl", 0.45))
        self.w_stability = float(w.get("stability", 0.15))
        self.w_risk      = float(w.get("risk", 0.20))
        self.w_momentum  = float(w.get("momentum", 0.10))
        self.w_sharpe    = float(w.get("sharpe", 0.05))
        self.w_rlfb      = float(w.get("rl_feedback", 0.05))  # RL feedback

        # Histories
        self.pnl_hist   = deque(maxlen=self.window)
        self.risk_hist  = deque(maxlen=self.window)  # vol + dd
        self.ret_hist   = deque(maxlen=self.window)  # price/portfolio returns for momentum/Sharpe

        # RL feedback scaling
        rl_cfg = fusion_cfg.get("rl_feedback", {})
        self.rl_key    = rl_cfg.get("key", "policy_confidence")  # e.g. 'advantage' or 'policy_confidence'
        self.rl_scale  = float(rl_cfg.get("scale", 1.0))
        self.rl_clip   = float(rl_cfg.get("clip", 1.0))          # clamp |feedback| ≤ clip

        # Momentum config (simple rolling mean of recent returns)
        mom_cfg = fusion_cfg.get("momentum", {})
        self.mom_mode   = mom_cfg.get("mode", "simple")          # 'simple' or 'ema'
        self.mom_alpha  = float(mom_cfg.get("alpha", 0.2))       # for EMA

        # Sharpe config
        sharpe_cfg = fusion_cfg.get("sharpe", {})
        self.sharpe_floor = float(sharpe_cfg.get("floor", 0.0))  # prevent extreme negatives

        self._ema_momentum = 0.0
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    def step(self, info: dict) -> float:
        """
        Compute fused reward using:
          - mean PnL
          - stability = 1 / (1 + mean_risk)
          - risk = average(volatility + drawdown)
          - momentum = recent returns (simple mean or EMA)
          - sharpe ≈ mean(returns) / std(returns)
          - rl_feedback = scaled term from info[self.rl_key]
        The result is clipped to [-1, 1].
        """
        pnl = float(info.get("pnl", 0.0))
        vol = abs(float(info.get("volatility", 0.0)))
        dd  = abs(float(info.get("drawdown", 0.0)))

        # Accept either portfolio return or primary symbol return
        ret = float(info.get("portfolio_return", info.get("return", 0.0)))

        # Update histories
        self.pnl_hist.append(pnl)
        self.risk_hist.append(vol + dd)
        self.ret_hist.append(ret)

        mean_pnl  = _safe_mean(self.pnl_hist)
        mean_risk = _safe_mean(self.risk_hist)
        stability = 1.0 / (1.0 + max(0.0, mean_risk))  # [0,1] with diminishing returns

        # Momentum
        if self.mom_mode.lower() == "ema":
            self._ema_momentum = self.mom_alpha * ret + (1 - self.mom_alpha) * self._ema_momentum
            momentum = self._ema_momentum
        else:
            momentum = _safe_mean(self.ret_hist)

        # Sharpe-like boost (safe, floor-protected denominator)
        mu = _safe_mean(self.ret_hist)
        sigma = _safe_std(self.ret_hist)
        denom = max(self.sharpe_floor, sigma)
        sharpe_like = mu / denom if denom > 0 else 0.0

        # RL Feedback (advantage or policy confidence)
        rl_feedback_raw = float(info.get(self.rl_key, 0.0))
        rl_feedback = max(-self.rl_clip, min(self.rl_clip, rl_feedback_raw * self.rl_scale))

        fused = (
            self.w_pnl       * mean_pnl +
            self.w_stability * stability -
            self.w_risk      * mean_risk +
            self.w_momentum  * momentum +
            self.w_sharpe    * sharpe_like +
            self.w_rlfb      * rl_feedback
        )

        fused = float(np.clip(fused, -1.0, 1.0))
        self.last_reward = fused

        logger.debug(
            "💠 RewardFusion -> "
            f"pnl={mean_pnl:.4f}, risk={mean_risk:.4f}, stab={stability:.4f}, "
            f"mom={momentum:.4f}, sh={sharpe_like:.4f}, rlfb={rl_feedback:.4f}, "
            f"fused={fused:.4f}"
        )
        return fused

    def last(self) -> float:
        return self.last_reward
