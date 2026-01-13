import math
from collections import deque
from typing import Dict

class RollingATRProxy:
    """
    ATR proxy using a rolling mean of absolute price changes (no H/L).
    ATR ≈ mean(|p_t - p_{t-1}|) over window. Good enough for sizing caps.
    """
    def __init__(self, window: int = 14):
        self.window = int(window)
        self._buf: Dict[str, deque] = {}

    def update(self, prices: Dict[str, float]) -> Dict[str, float]:
        atr = {}
        for sym, p in prices.items():
            dq = self._buf.setdefault(sym, deque(maxlen=self.window))
            if dq and dq[-1] is not None:
                change = abs(p - dq[-1])
                dq.append(p)
                # compute mean abs change
                atr[sym] = (sum(abs(dq[i] - dq[i-1]) for i in range(1, len(dq))) / max(1, len(dq)-1)) if len(dq) >= 2 else 0.0
            else:
                dq.append(p)
                atr[sym] = 0.0
        return atr

    def current(self, symbol: str) -> float:
        dq = self._buf.get(symbol)
        if not dq or len(dq) < 2:
            return 0.0
        return sum(abs(dq[i] - dq[i-1]) for i in range(1, len(dq))) / max(1, len(dq)-1)


class PreTradeRisk:
    """
    Computes max allowed quantity for a trade BEFORE routing, based on:
      - equity * max_trade_risk (risk $ budget)
      - per-share risk ≈ ATR_proxy * atr_multiplier
    """

    def __init__(self, cfg):
        r = cfg.get("risk_aware", {})
        self.max_trade_risk = float(r.get("max_trade_risk", 0.01))
        self.atr_mult = float(r.get("atr_multiplier", 1.5))
        self.atr_proxy = RollingATRProxy(window=int(r.get("atr_window", 14)))

    def update_prices(self, prices: Dict[str, float]) -> Dict[str, float]:
        return self.atr_proxy.update(prices)

    def max_qty_for(self, symbol: str, equity: float, price: float) -> int:
        risk_budget = float(equity) * self.max_trade_risk  # $ we can risk
        atr = self.atr_proxy.current(symbol)
        per_share_risk = max(1e-4, atr * self.atr_mult)   # $/share we’re risking
        # if ATR proxy is too small (early), fall back to a tiny percentage of price
        if per_share_risk < 1e-4:
            per_share_risk = max(0.0025 * price, 0.05)
        max_shares = int(risk_budget / per_share_risk)
        return max(max_shares, 0)
