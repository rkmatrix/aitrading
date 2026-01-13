# ai/risk/symbol_micro_allocator.py
# ---------------------------------------------------------------
# Phase 27 – Symbol Micro-Allocator
#
# Provides per-symbol adaptive sizing, cooldown, volatility/risk
# scaling, win/loss streak intelligence, RL reward biasing,
# and per-symbol drawdown penalty.
#
# ---------------------------------------------------------------

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger("SymbolMicroAllocator")


# ================================================================
# Dataclass: Per-symbol tracked stats
# ================================================================
@dataclass
class SymbolStats:
    symbol: str

    wins: int = 0
    losses: int = 0
    consec_wins: int = 0
    consec_losses: int = 0

    realized_pnl: float = 0.0

    # Worst drawdown (fraction of equity) seen for this symbol
    max_drawdown_frac: float = 0.0

    # RL reward statistics
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    reward_count: int = 0

    # Cooldown ticks after large losses
    cooldown_ticks_remaining: int = 0

    # -------------------------------------------
    # Derived properties
    # -------------------------------------------
    @property
    def avg_reward(self) -> float:
        if self.reward_count <= 0:
            return 0.0
        return self.reward_sum / self.reward_count

    @property
    def reward_volatility(self) -> float:
        if self.reward_count <= 1:
            return 0.0
        mean = self.reward_sum / self.reward_count
        mean_sq = self.reward_sq_sum / self.reward_count
        variance = max(mean_sq - mean * mean, 0.0)
        return variance ** 0.5

    # -------------------------------------------
    # Update functions
    # -------------------------------------------
    def register_reward(self, reward: float) -> None:
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward
        self.reward_count += 1

    def register_trade_result(self, pnl_dollars: float, equity_at_trade: float) -> None:
        """Update win/loss and drawdown tracking."""
        self.realized_pnl += pnl_dollars

        # Win / loss streak
        if pnl_dollars > 0:
            self.wins += 1
            self.consec_wins += 1
            self.consec_losses = 0
        elif pnl_dollars < 0:
            self.losses += 1
            self.consec_losses += 1
            self.consec_wins = 0

            if equity_at_trade > 0:
                dd = abs(pnl_dollars) / equity_at_trade
                if dd > self.max_drawdown_frac:
                    self.max_drawdown_frac = dd

    # -------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolStats":
        return cls(**d)


# ================================================================
# Dataclass: YAML config → runtime settings
# ================================================================
@dataclass
class MicroAllocConfig:
    enabled: bool = True

    base_max_pct: float = 0.03
    global_aggression: float = 1.0

    streak_scale_factor: float = 0.15
    vol_scale_factor: float = 0.20
    dd_penalty_factor: float = 0.30
    reward_scale_factor: float = 0.10

    loss_cooldown_ticks: int = 5
    loss_cooldown_equity_pct: float = 0.01  # 1% equity loss triggers cooldown

    min_max_pct: float = 0.005
    max_max_pct: float = 0.08

    state_path: Optional[Path] = None

    log_level: str = "INFO"

    # -------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MicroAllocConfig":
        raw = dict(raw or {})
        p = raw.get("state_path")
        raw["state_path"] = Path(p) if p else None
        return cls(**raw)


# ================================================================
# Main class: SymbolMicroAllocator
# ================================================================
class SymbolMicroAllocator:
    """
    Main Phase 27 module.

    Responsibilities:
    - Track per-symbol stats
    - Produce per-symbol max_position_pct, multiplier, cooldown
    - Handle volatility → size transform
    - Handle streak → size transform
    - Handle drawdown → size transform
    - Handle RL reward → size transform
    """

    def __init__(self, config: MicroAllocConfig) -> None:
        self.cfg = config
        self.stats: Dict[str, SymbolStats] = {}

        logger.setLevel(getattr(logging, self.cfg.log_level.upper(), logging.INFO))

        self._load_state_if_exists()

        logger.info(
            "SymbolMicroAllocator initialized (enabled=%s, base_max_pct=%.4f)",
            self.cfg.enabled,
            self.cfg.base_max_pct,
        )

    # ============================================================
    # State Persistence
    # ============================================================
    def _load_state_if_exists(self) -> None:
        if not self.cfg.state_path:
            return
        try:
            if self.cfg.state_path.exists():
                raw = json.loads(self.cfg.state_path.read_text(encoding="utf-8"))
                for sym, sd in raw.get("symbols", {}).items():
                    self.stats[sym] = SymbolStats.from_dict(sd)
                logger.info(
                    "Loaded SymbolMicroAllocator state for %d symbols",
                    len(self.stats),
                )
        except Exception:
            logger.exception("Failed to load micro allocator state.")

    def save_state(self) -> None:
        if not self.cfg.state_path:
            return
        try:
            payload = {"symbols": {s: st.to_dict() for s, st in self.stats.items()}}
            self.cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.state_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            logger.exception("Failed to save micro allocator state.")

    # ============================================================
    # Factory method
    # ============================================================
    @classmethod
    def from_yaml(cls, yaml_cfg: Dict[str, Any]) -> "SymbolMicroAllocator":
        config = MicroAllocConfig.from_dict(yaml_cfg)
        return cls(config)

    # ============================================================
    # Public API used by Phase 26
    # ============================================================
    def on_tick(self) -> None:
        """Decay cooldown timers each tick."""
        for st in self.stats.values():
            if st.cooldown_ticks_remaining > 0:
                st.cooldown_ticks_remaining -= 1

    def register_symbol_observation(
        self,
        symbol: str,
        *,
        volatility: Optional[float] = None,
        symbol_drawdown: Optional[float] = None,
        rl_reward: Optional[float] = None,
    ) -> None:
        st = self._get_or_create(symbol)

        if symbol_drawdown is not None:
            st.max_drawdown_frac = max(st.max_drawdown_frac, max(symbol_drawdown, 0.0))

        if rl_reward is not None:
            st.register_reward(rl_reward)

    def register_trade_result(
        self, symbol: str, *, pnl_dollars: float, equity_at_trade: float
    ) -> None:
        st = self._get_or_create(symbol)
        st.register_trade_result(pnl_dollars, equity_at_trade)

        # Apply cooldown if loss > equity threshold
        if pnl_dollars < 0 and equity_at_trade > 0:
            loss_frac = abs(pnl_dollars) / equity_at_trade
            if loss_frac >= self.cfg.loss_cooldown_equity_pct:
                st.cooldown_ticks_remaining = max(
                    st.cooldown_ticks_remaining, self.cfg.loss_cooldown_ticks
                )
                logger.warning(
                    "Symbol %s entering cooldown for %d ticks (loss_frac %.4f)",
                    symbol,
                    st.cooldown_ticks_remaining,
                    loss_frac,
                )

        self.save_state()

    # ============================================================
    # Core sizing output
    # ============================================================
    def get_constraints(
        self,
        symbol: str,
        *,
        equity: float,
        volatility: Optional[float],
        symbol_drawdown: Optional[float],
        rl_reward: Optional[float],
        base_target_dollars: Optional[float],
    ) -> Dict[str, Any]:
        st = self._get_or_create(symbol)

        if not self.cfg.enabled or equity <= 0:
            # simple fallback
            return {
                "symbol": symbol,
                "max_position_pct": self.cfg.base_max_pct,
                "position_multiplier": 1.0,
                "cooldown_active": False,
                "base_qty": 0.0,
            }

        cooldown_active = st.cooldown_ticks_remaining > 0

        # --------------------------------------------------------
        # Build multipliers
        # --------------------------------------------------------
        base_max = self.cfg.base_max_pct * max(self.cfg.global_aggression, 0.0)

        streak_mult = self._calc_streak_mult(st)
        vol_mult = self._calc_vol_mult(volatility)
        dd_mult = self._calc_dd_mult(symbol_drawdown, st.max_drawdown_frac)
        reward_mult = self._calc_reward_mult(
            rl_reward, st.avg_reward, st.reward_volatility
        )

        position_multiplier = streak_mult * vol_mult * dd_mult * reward_mult

        # If cooling down → kill size completely
        if cooldown_active:
            position_multiplier = 0.0

        # Clamp symbol-specific max%
        max_position_pct = max(
            self.cfg.min_max_pct,
            min(self.cfg.max_max_pct, base_max * max(position_multiplier, 0.0)),
        )

        # Compute auto base qty (optional)
        base_qty = 0.0
        if base_target_dollars is not None and equity > 0:
            base_qty = (base_target_dollars * position_multiplier) / equity

        return {
            "symbol": symbol,
            "max_position_pct": max_position_pct,
            "position_multiplier": position_multiplier,
            "cooldown_active": cooldown_active,
            "base_qty": base_qty,
        }

    # ============================================================
    # Internal helpers
    # ============================================================
    def _get_or_create(self, symbol: str) -> SymbolStats:
        if symbol not in self.stats:
            self.stats[symbol] = SymbolStats(symbol=symbol)
        return self.stats[symbol]

    # -------------------------------------------
    def _calc_streak_mult(self, st: SymbolStats) -> float:
        f = max(self.cfg.streak_scale_factor, 0.0)
        if f == 0:
            return 1.0

        if st.consec_wins > 0:
            return 1.0 + f * min(st.consec_wins, 5)
        if st.consec_losses > 0:
            return max(0.25, 1.0 - f * min(st.consec_losses, 5))
        return 1.0

    # -------------------------------------------
    def _calc_vol_mult(self, vol: Optional[float]) -> float:
        f = max(self.cfg.vol_scale_factor, 0.0)
        if f == 0 or vol is None:
            return 1.0

        v = max(float(vol), 0.0)

        # For v <= 1.0 → mild boost
        if v <= 1.0:
            bonus = f * (1.0 - v)
            return max(0.5, 1.0 + min(bonus, 0.5))

        # For v > 1.0 → shrink
        penalty = f * (v - 1.0)
        return max(0.25, 1.0 / (1.0 + penalty))

    # -------------------------------------------
    def _calc_dd_mult(self, dd_now: Optional[float], dd_hist: float) -> float:
        f = max(self.cfg.dd_penalty_factor, 0.0)
        if f == 0:
            return 1.0

        d = max(dd_now or 0.0, dd_hist, 0.0)
        d = min(d, 0.5)  # cap DD at 50%
        penalty = f * (d / 0.5)
        return max(0.4, 1.0 - penalty)

    # -------------------------------------------
    def _calc_reward_mult(
        self, reward_now: Optional[float], avg_reward: float, reward_vol: float
    ) -> float:
        f = max(self.cfg.reward_scale_factor, 0.0)
        if f == 0:
            return 1.0

        effective = reward_now if reward_now is not None else avg_reward

        if reward_vol > 0:
            score = effective / (reward_vol + 1e-8)
        else:
            score = effective

        score = max(min(score, 3.0), -3.0)
        mult = 1.0 + f * (score / 3.0)
        return max(0.5, min(1.5, mult))
