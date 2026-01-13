from typing import Dict
import numpy as np
import pandas as pd
from ai.metrics.metrics import summary_stats


def evaluate_policy_on_window(
    features: pd.DataFrame,
    policy_predict_fn,
    price_col: str = "close",
    fee_bps: float = 1.0,
) -> Dict:
    """
    Lightweight evaluation: walk forward through features with the current policy,
    translate actions {-1,0,1} to position, compute per-bar PnL net of fees.
    """
    if features is None or len(features) < 5:
        return {"pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0}

    prices = features[price_col].astype(float)
    pos = 0
    last_price = float(prices.iloc[0])
    fee = fee_bps / 1e4
    rets = []

    for i in range(1, len(features)):
        obs = features.iloc[i - 1]  # we pretend obs is the feature row; policy wrapper will ignore extra fields if needed
        action = int(policy_predict_fn(obs))
        target_pos = [-1, 0, 1][max(0, min(2, action))]
        p = float(prices.iloc[i])

        # transaction fee if changing position
        trade_cost = 0.0
        if target_pos != pos:
            trade_cost = p * fee * abs(target_pos - pos)

        # PnL from holding previous bar
        pnl = (p - last_price) * pos - trade_cost
        rets.append(pnl / max(1.0, last_price))  # scaled return

        pos = target_pos
        last_price = p

    rets = pd.Series(rets, index=features.index[1:])
    return summary_stats(rets, start_value=1.0)
