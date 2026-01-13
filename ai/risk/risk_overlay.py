from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd
from datetime import time as dtime


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.15         # halt if equity falls this % from peak
    vola_lookback: int = 30                # bars for realized vol
    vola_cap: float = 0.04                 # dailyized vol cap (approx; interpret per interval)
    market_open: str = "09:30"
    market_close: str = "16:00"
    force_flat_outside_market: bool = True
    min_equity: float = 1000.0             # safety floor


class RiskOverlay:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.equity_peak = None
        self.halted = False

    def reset(self, starting_equity: float):
        self.equity_peak = starting_equity
        self.halted = False

    def _is_market_open(self, ts_utc: pd.Timestamp) -> bool:
        if not isinstance(ts_utc, pd.Timestamp):
            return True
        # Convert to US/Eastern for hours guard (approx; adjust if you track timezone precisely)
        try:
            ts_et = ts_utc.tz_convert("US/Eastern")
        except Exception:
            ts_et = ts_utc
        open_h, open_m = map(int, self.cfg.market_open.split(":"))
        close_h, close_m = map(int, self.cfg.market_close.split(":"))
        t = ts_et.time()
        return dtime(open_h, open_m) <= t <= dtime(close_h, close_m)

    def _realized_vol(self, close_series: pd.Series) -> float:
        if close_series is None or len(close_series) < self.cfg.vola_lookback + 1:
            return 0.0
        rets = close_series.pct_change().dropna()
        if rets.empty:
            return 0.0
        # “Dailyized” rough scaling: not perfect for intraday, but acts as a consistent guard.
        # For 1m bars, 390 bars ≈ 1 trading day. For 5m bars, ~78 bars/day, etc.
        bars_per_day = max(1, int(390 / max(1, int(len(close_series) / 30))))
        vol = float(rets.tail(self.cfg.vola_lookback).std()) * np.sqrt(bars_per_day)
        return vol

    def advise(self, now_ts: pd.Timestamp, equity: float, prices: pd.Series) -> Dict:
        """
        Returns decisions dict:
        {
          "force_flat": bool,    # true if risk wants no position
          "halt": bool,          # true if trading should halt fully
          "reason": str,         # human readable
        }
        """
        if self.equity_peak is None:
            self.equity_peak = equity
        self.equity_peak = max(self.equity_peak, equity)

        # Drawdown guard
        if self.equity_peak > 0:
            dd = 1.0 - (equity / self.equity_peak)
            if dd >= self.cfg.max_drawdown_pct:
                self.halted = True
                return {"force_flat": True, "halt": True, "reason": f"Max DD {dd:.1%} >= {self.cfg.max_drawdown_pct:.1%}"}

        # Equity floor
        if equity <= self.cfg.min_equity:
            self.halted = True
            return {"force_flat": True, "halt": True, "reason": f"Equity floor {equity:.2f} <= {self.cfg.min_equity:.2f}"}

        # Volatility guard
        vol = self._realized_vol(prices)
        if vol >= self.cfg.vola_cap:
            return {"force_flat": True, "halt": False, "reason": f"Vol cap {vol:.2%} >= {self.cfg.vola_cap:.2%}"}

        # Market hours guard
        if self.cfg.force_flat_outside_market and not self._is_market_open(now_ts):
            return {"force_flat": True, "halt": False, "reason": "Outside market hours"}

        # Default: allow trading
        return {"force_flat": False, "halt": False, "reason": ""}


def simple_position_sizer(target_pos: int, base_qty: int, equity: float, risk_per_trade_pct: float = 0.01) -> int:
    """
    Very simple sizer: scales base_qty up to a cap proportional to equity.
    """
    if target_pos == 0:
        return 0
    cap_qty = max(1, int((equity * risk_per_trade_pct) / 100.0))
    return int(max(1, min(cap_qty, base_qty)))
