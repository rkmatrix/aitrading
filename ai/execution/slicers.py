# ai/execution/slicers.py
# ---------------------------------------------------------------------
# Phase 9.4 — Execution Quality: order slicing engines (VWAP/TWAP)
#
# Produces a schedule of child "clips" (qty + target time window).
# The TradeExecutor will actually submit/simulate fills against quotes.
#
# Author: AITradeBot

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import datetime as dt


@dataclass
class SliceClip:
    symbol: str
    side: str            # 'BUY' or 'SELL'
    qty: float           # shares (can be fractional if supported)
    t_start: dt.datetime
    t_end: dt.datetime


class BaseSlicer:
    def plan(self, *, symbol: str, side: str, qty: float,
             start: dt.datetime, horizon_sec: int, n_slices: int) -> List[SliceClip]:
        raise NotImplementedError


class TWAPSlicer(BaseSlicer):
    """Evenly splits qty into n_slices over the horizon — simple & robust."""

    def plan(self, *, symbol: str, side: str, qty: float,
             start: dt.datetime, horizon_sec: int, n_slices: int) -> List[SliceClip]:
        n = max(1, n_slices)
        per = float(qty) / n
        clips: List[SliceClip] = []
        for i in range(n):
            t0 = start + dt.timedelta(seconds=int(i * horizon_sec / n))
            t1 = start + dt.timedelta(seconds=int((i + 1) * horizon_sec / n))
            clips.append(SliceClip(symbol=symbol, side=side, qty=per, t_start=t0, t_end=t1))
        return clips


class VWAPSlicer(BaseSlicer):
    """
    Approximates intrahorizon volume curve as a symmetric hump (Gaussian-ish),
    allocating larger clips near the middle of the horizon.
    """

    def plan(self, *, symbol: str, side: str, qty: float,
             start: dt.datetime, horizon_sec: int, n_slices: int) -> List[SliceClip]:
        n = max(1, n_slices)
        # Create a bell curve over [0,1]
        x = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
        curve = np.exp(-0.5 * ((x - 0.5) / 0.22) ** 2)  # stdev ~0.22 gives nice center weight
        weights = curve / curve.sum()
        clips: List[SliceClip] = []
        for i, w in enumerate(weights):
            q = float(qty) * float(w)
            t0 = start + dt.timedelta(seconds=int(i * horizon_sec / n))
            t1 = start + dt.timedelta(seconds=int((i + 1) * horizon_sec / n))
            clips.append(SliceClip(symbol=symbol, side=side, qty=q, t_start=t0, t_end=t1))
        return clips


def get_slicer(kind: str) -> BaseSlicer:
    kind = (kind or "TWAP").upper()
    if kind == "VWAP":
        return VWAPSlicer()
    return TWAPSlicer()
