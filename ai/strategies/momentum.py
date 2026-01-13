# ==========================================================
# ai/strategies/momentum.py
# ==========================================================
# Momentum Strategy Module (TA library)
# Handles yfinance 0.2.40+ DataFrame shapes automatically
# ==========================================================

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator


def momentum_score(df: pd.DataFrame) -> float:
    """
    Compute normalized momentum score in [-1, +1].

    Formula:
        rsi_component = (RSI - 50) / 50
        ema_component = (EMA12 - EMA26) / EMA26
        score = 0.6*rsi_component + 0.4*ema_component
    """
    if df is None or len(df) < 30:
        return 0.0

    try:
        # --- Normalize close series shape ---
        close = df["close"] if "close" in df.columns else df.iloc[:, 3]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # flatten (N,1) → (N,)
        close = pd.Series(close).astype(float)

        # --- Indicators ---
        rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
        ema_fast = EMAIndicator(close, window=12).ema_indicator().iloc[-1]
        ema_slow = EMAIndicator(close, window=26).ema_indicator().iloc[-1]

        # --- Components ---
        rsi_component = (rsi - 50) / 50
        ema_component = (ema_fast - ema_slow) / ema_slow

        score = 0.6 * rsi_component + 0.4 * ema_component
        return float(max(-1.0, min(1.0, score)))

    except Exception as e:
        print(f"⚠️  momentum_score failed: {e}")
        return 0.0


if __name__ == "__main__":
    import yfinance as yf

    print("📊 Testing momentum_score on AAPL...")
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    df = df.rename(columns=str.lower)
    print(f"Momentum score (AAPL): {momentum_score(df):.3f}")
