from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import yaml  # type: ignore
import numpy as np  # type: ignore

logger = logging.getLogger(__name__)

try:
    from tools.telegram_alerts import notify  # type: ignore
except Exception:
    notify = None  # type: ignore


@dataclass
class AggressionConfig:
    min_factor: float = 0.5
    max_factor: float = 1.5
    chill_dd_pct: float = 0.02
    caution_dd_pct: float = 0.05
    defense_dd_pct: float = 0.10


@dataclass
class SlopeConfig:
    window: int = 20
    boost_threshold: float = 0.001
    cut_threshold: float = -0.001


@dataclass
class TelegramConfig:
    enable: bool = True
    tag: str = "phase69e_equity_agent"


@dataclass
class EquityAgentConfig:
    mode: str = "PAPER"
    aggression: AggressionConfig = field(default_factory=AggressionConfig)
    slope: SlopeConfig = field(default_factory=SlopeConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


class EquityGrowthAgent:
    """
    Regime-aware agent that outputs aggression_factor based on equity curve slope + drawdown.
    """

    def __init__(self, cfg: EquityAgentConfig) -> None:
        self.cfg = cfg
        self._last_regime: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "EquityGrowthAgent":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        a = raw.get("aggression", {}) or {}
        s = raw.get("slope", {}) or {}
        t = raw.get("telegram", {}) or {}

        ag = AggressionConfig(
            min_factor=float(a.get("min_factor", 0.5)),
            max_factor=float(a.get("max_factor", 1.5)),
            chill_dd_pct=float(a.get("chill_dd_pct", 0.02)),
            caution_dd_pct=float(a.get("caution_dd_pct", 0.05)),
            defense_dd_pct=float(a.get("defense_dd_pct", 0.10)),
        )

        sl = SlopeConfig(
            window=int(s.get("window", 20)),
            boost_threshold=float(s.get("boost_threshold", 0.001)),
            cut_threshold=float(s.get("cut_threshold", -0.001)),
        )

        tel = TelegramConfig(
            enable=bool(t.get("enable", True)),
            tag=str(t.get("tag", "phase69e_equity_agent")),
        )

        cfg = EquityAgentConfig(
            mode=str(raw.get("mode", "PAPER")),
            aggression=ag,
            slope=sl,
            telegram=tel,
        )

        return cls(cfg)

    # ---------------------------------------------------------

    def _compute_slope(self, curve: List[float]) -> float:
        if len(curve) < 2:
            return 0.0

        n = min(len(curve), self.cfg.slope.window)
        y = np.array(curve[-n:], dtype=float)
        x = np.arange(n, dtype=float)

        x_mean = x.mean()
        y_mean = y.mean()

        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()

        if den <= 0:
            return 0.0

        slope = num / den
        return float(slope)

    # ---------------------------------------------------------

    def step(
        self,
        equity_curve: List[float],
        intraday_drawdown_pct: float,
    ) -> Tuple[float, Dict[str, Any]]:
        if not equity_curve:
            return 1.0, {"regime": "neutral", "reason": "no_equity_history"}

        slope = self._compute_slope(equity_curve)
        dd = intraday_drawdown_pct

        base = 1.0
        regime = "neutral"
        reason = ""

        if dd >= self.cfg.aggression.defense_dd_pct:
            regime = "hard_defense"
            factor = self.cfg.aggression.min_factor
            reason = f"drawdown {dd:.2%} >= defense {self.cfg.aggression.defense_dd_pct:.2%}"
        elif dd >= self.cfg.aggression.caution_dd_pct:
            regime = "caution"
            factor = max(self.cfg.aggression.min_factor, base * 0.7)
            reason = f"drawdown {dd:.2%} >= caution {self.cfg.aggression.caution_dd_pct:.2%}"
        elif dd >= self.cfg.aggression.chill_dd_pct:
            regime = "chill"
            factor = max(self.cfg.aggression.min_factor, base * 0.9)
            reason = f"drawdown {dd:.2%} >= chill {self.cfg.aggression.chill_dd_pct:.2%}"
        else:
            if slope >= self.cfg.slope.boost_threshold:
                regime = "attack"
                factor = min(self.cfg.aggression.max_factor, base * 1.3)
                reason = f"slope {slope:.6f} >= boost {self.cfg.slope.boost_threshold:.6f}"
            elif slope <= self.cfg.slope.cut_threshold:
                regime = "soft_defense"
                factor = max(self.cfg.aggression.min_factor, base * 0.8)
                reason = f"slope {slope:.6f} <= cut {self.cfg.slope.cut_threshold:.6f}"
            else:
                regime = "neutral"
                factor = base
                reason = "slope in neutral range"

        factor = max(self.cfg.aggression.min_factor, min(self.cfg.aggression.max_factor, factor))

        if regime != self._last_regime:
            self._last_regime = regime
            self._alert(
                f"🎯 Equity regime → {regime} | factor={factor:.2f} | slope={slope:.6f} | dd={dd:.2%}"
            )

        meta = {
            "regime": regime,
            "factor": factor,
            "slope": slope,
            "drawdown": dd,
            "reason": reason,
        }
        return factor, meta

    # ---------------------------------------------------------

    def _alert(self, msg: str) -> None:
        if not self.cfg.telegram.enable or notify is None:
            logger.info("[EquityAgent] %s", msg)
            return

        try:
            notify(msg, kind="pnl", meta={"tag": self.cfg.telegram.tag, "mode": self.cfg.mode})
        except Exception:
            logger.exception("Failed to send equity agent alert")
