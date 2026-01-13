# ai/eval/self_evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import math

@dataclass
class EvalEvent:
    pnl: float
    volatility: float
    regime: str   # 'bull'|'bear'|'sideways'
    win: bool

@dataclass
class EvalSummary:
    pnl: float
    sharpe_like: float    # mean/std of pnl stream (eps for stability)
    winrate: float
    trades: int
    vol_avg: float

class SelfEvaluator:
    """
    Streaming evaluator: feed each fill/event, call summary() anytime.
    Uses a robust SR proxy over the event PnL stream.
    """
    def __init__(self):
        self._ev: List[EvalEvent] = []

    def add_event(self, pnl: float, volatility: float, regime: str):
        self._ev.append(EvalEvent(pnl=float(pnl), volatility=float(volatility), regime=str(regime), win=(pnl > 0)))

    def reset(self):
        self._ev.clear()

    def summary(self) -> EvalSummary:
        if not self._ev:
            return EvalSummary(0.0, 0.0, 0.0, 0, 0.0)
        pnls = [e.pnl for e in self._ev]
        n = len(pnls)
        mean = sum(pnls)/n
        var = sum((x-mean)**2 for x in pnls)/max(1, n-1)
        std = math.sqrt(var + 1e-12)
        sr_like = mean/max(std, 1e-6)
        winrate = sum(1 for e in self._ev if e.win)/n
        vol_avg = sum(e.volatility for e in self._ev)/n
        return EvalSummary(pnl=sum(pnls), sharpe_like=sr_like, winrate=winrate, trades=n, vol_avg=vol_avg)
