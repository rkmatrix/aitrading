# ai/reward/volatility_metrics.py
from __future__ import annotations
import numpy as np

def safe_array(x) -> np.ndarray:
    if x is None:
        return np.asarray([])
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr

def realized_vol(returns, window: int = 30, ddof: int = 1) -> float:
    r = safe_array(returns)
    if r.size == 0:
        return 0.0
    r = r[-window:]
    if r.size < 2:
        return 0.0
    return float(np.sqrt(np.var(r, ddof=ddof)))

def ewma_vol(returns, span: int = 20) -> float:
    r = safe_array(returns)
    if r.size == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    mean = 0.0
    var = 0.0
    for x in r:
        delta = x - mean
        mean += alpha * delta
        var = (1 - alpha) * (var + alpha * delta * delta)
    return float(np.sqrt(max(var, 0.0)))

def max_drawdown(equity_curve) -> float:
    ec = safe_array(equity_curve)
    if ec.size == 0:
        return 0.0
    run_max = np.maximum.accumulate(ec)
    drawdown = (ec - run_max) / np.maximum(run_max, 1e-9)
    return float(np.min(drawdown))  # negative number (e.g., -0.23)

def sharpe_like(returns, risk_free: float = 0.0) -> float:
    r = safe_array(returns)
    if r.size == 0:
        return 0.0
    ex = r - risk_free
    vol = realized_vol(ex, window=len(ex), ddof=1)
    if vol <= 1e-12:
        return 0.0
    return float(np.mean(ex) / vol)

def parkinson_vol(high, low, window: int = 30) -> float:
    h = safe_array(high)
    l = safe_array(low)
    n = min(h.size, l.size, window)
    if n < 2:
        return 0.0
    h = h[-n:]
    l = l[-n:]
    log_hl = np.log(np.maximum(h, 1e-9)) - np.log(np.maximum(l, 1e-9))
    factor = 1.0 / (4.0 * np.log(2.0))
    return float(np.sqrt(factor * np.mean(log_hl ** 2)))
