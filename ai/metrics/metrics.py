from typing import Dict
import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    returns = returns.fillna(0.0)
    curve = (1.0 + returns).cumprod() * start_value
    return curve


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, scale: int = 252) -> float:
    # For intraday, scale can be ~252 (daily) if returns are daily; adjust if per-bar
    r = returns - risk_free / scale
    std = r.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(scale) * r.mean() / std)


def max_drawdown(curve: pd.Series) -> float:
    if curve is None or len(curve) == 0:
        return 0.0
    roll_max = curve.cummax()
    dd = (curve / roll_max) - 1.0
    return float(dd.min())


def hit_rate(returns: pd.Series) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    wins = (returns > 0).sum()
    return float(wins) / float(len(returns))


def summary_stats(returns: pd.Series, start_value: float = 1.0) -> Dict:
    curve = equity_curve(returns, start_value=start_value)
    stats = {
        "pnl": float(curve.iloc[-1] - start_value) if len(curve) else 0.0,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(curve),
        "hit_rate": hit_rate(returns),
    }
    return stats
