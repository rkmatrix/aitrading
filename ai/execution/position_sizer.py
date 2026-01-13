# ai/execution/position_sizer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import math

@dataclass
class SizeConfig:
    # Core risk budget
    risk_budget_daily: float = 0.01         # 1% of equity per day
    risk_budget_trade: float = 0.003        # 30 bps per trade max VaR

    # Vol targeting
    vol_floor: float = 1e-4                 # avoid div-by-zero explosions
    target_horizon_steps: int = 20          # “1-day” if step≈1/20 day
    kelly_fraction: float = 0.25            # Kelly-lite shrink

    # Caps & floors
    max_gross_leverage: float = 2.0
    max_position_per_asset: float = 0.5     # 50% of equity (abs)
    min_notional: float = 100.0             # $ min trade
    lot_step: float = 1.0                   # shares step (or contracts)

    # Smoothing (to reduce churn)
    exposure_ema_alpha: float = 0.3         # 0..1 (higher = faster adjust)

    # Optional aggressiveness scaling from Phase 33's risk_mult
    use_risk_multiplier: bool = True
    risk_mult_floor: float = 0.5
    risk_mult_ceiling: float = 1.5

@dataclass
class PortfolioState:
    equity: float
    price_map: Dict[str, float]             # symbol -> last price

class VolAdaptiveSizer:
    """
    Converts a direction signal into a *size* (shares) per asset, using:
      size ≈ kelly_frac * risk_budget / (vol * sqrt(horizon))
    Then applies caps, min_notional, and exposure EMA smoothing.
    """
    def __init__(self, cfg: SizeConfig):
        self.cfg = cfg
        self._ema_exposure: Dict[str, float] = {}  # symbol -> last target exposure [-1..+1]

    def _clip(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _risk_scale(self, risk_mult: Optional[float]) -> float:
        if not self.cfg.use_risk_multiplier or risk_mult is None:
            return 1.0
        return self._clip(risk_mult, self.cfg.risk_mult_floor, self.cfg.risk_mult_ceiling)

    def target_exposure(
        self,
        symbol: str,
        direction: float,               # -1..+1
        vol: Optional[float],
        equity: float,
        risk_mult: Optional[float] = None
    ) -> float:
        """
        Returns target exposure in [-max_pos_per_asset, +max_pos_per_asset] as fraction of equity.
        If vol is None, treat as high vol (smaller size).
        """
        v = max(vol or 10 * self.cfg.vol_floor, self.cfg.vol_floor)
        horizon = max(self.cfg.target_horizon_steps, 1)
        # Vol targeting core
        base = (self.cfg.kelly_fraction *
                (self.cfg.risk_budget_trade / (v * math.sqrt(horizon))))
        # Apply direction and dynamic risk multiplier
        base *= float(direction)
        base *= self._risk_scale(risk_mult)

        # Cap by per-asset and overall leverage later
        base = self._clip(base, -self.cfg.max_position_per_asset, self.cfg.max_position_per_asset)
        return base

    def _ema(self, key: str, x: float) -> float:
        a = self.cfg.exposure_ema_alpha
        prev = self._ema_exposure.get(key, 0.0)
        y = a * x + (1 - a) * prev
        self._ema_exposure[key] = y
        return y

    def exposure_to_shares(self, symbol: str, exposure: float, state: PortfolioState) -> float:
        """
        exposure is fraction of equity allocated to this symbol (can be negative).
        Convert to *shares* using last price and lot step, enforce min_notional.
        """
        px = max(state.price_map.get(symbol, 0.0), 1e-6)
        notional = exposure * state.equity
        # Enforce notional min (unless exposure is ~0)
        if abs(notional) < self.cfg.min_notional:
            return 0.0
        shares = notional / px
        # Lot rounding
        step = max(self.cfg.lot_step, 1e-6)
        shares = round(shares / step) * step
        return shares

    def size_for_signal(
        self,
        symbol: str,
        direction: float,                 # -1..+1 (e.g., from discrete {-1,0,1})
        vol: Optional[float],
        state: PortfolioState,
        risk_mult: Optional[float] = None
    ) -> float:
        tgt_exp = self.target_exposure(symbol, direction, vol, state.equity, risk_mult=risk_mult)
        smoothed = self._ema(symbol, tgt_exp)
        return self.exposure_to_shares(symbol, smoothed, state)
