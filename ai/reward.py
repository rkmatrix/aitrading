import numpy as np
from dataclasses import dataclass

@dataclass
class RewardConfig:
    slippage_bps: float = 2.0
    fees_bps: float = 0.5
    dd_penalty: float = 2.0         # penalize drawdowns
    pos_penalty: float = 0.2        # penalize large positions vs equity
    overnight_penalty: float = 0.1  # inventory cost across sessions
    scale: float = 1.0              # overall scale of reward

class RewardEngine:
    """
    Computes reward at each decision step using mark-to-market PnL delta,
    minus penalties for risk & frictions. Returns scalar reward and components.
    """
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._peak_equity = None

    def reset(self, equity0: float):
        self._peak_equity = equity0

    def __call__(self, *, prev_equity: float, equity: float,
                 pos_value_abs: float, equity_base: float,
                 traded_notional: float, is_overnight_boundary: bool):
        # raw PnL delta
        pnl = equity - prev_equity

        # mark frictions on traded notional
        frictions = (self.cfg.slippage_bps + self.cfg.fees_bps) * 1e-4 * traded_notional

        # drawdown penalty
        self._peak_equity = max(self._peak_equity, equity) if self._peak_equity else equity
        dd = max(0.0, (self._peak_equity - equity) / self._peak_equity)
        dd_pen = self.cfg.dd_penalty * dd

        # position size penalty relative to equity (keeps positions modest)
        pos_frac = pos_value_abs / max(1e-9, equity_base)
        pos_pen = self.cfg.pos_penalty * (pos_frac ** 2)

        # overnight inventory cost
        overnight_pen = self.cfg.overnight_penalty if is_overnight_boundary and pos_value_abs > 0 else 0.0

        reward = (pnl - frictions) / max(1e-9, equity_base)  # normalize by equity
        reward -= (dd_pen + pos_pen + overnight_pen)
        return float(self.cfg.scale * reward), {
            "pnl": pnl, "frictions": frictions, "dd": dd,
            "pos_frac": pos_frac, "overnight": bool(overnight_pen)
        }
