# ==========================================================
# ai/strategies/mean_reversion.py
# ==========================================================
# Mean-Reversion Strategy Module (TA library)
# Handles yfinance 0.2.40+ shapes automatically
# ==========================================================

import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


def meanrev_score(df: pd.DataFrame) -> float:
    """
    Compute normalized mean-reversion score in [-1, +1].

    Logic:
        - Price below lower BB → +1 (buy)
        - Price above upper BB → -1 (sell)
        - RSI reinforces the signal
    """
    if df is None or len(df) < 30:
        return 0.0

    try:
        # --- Normalize close shape ---
        close = df["close"] if "close" in df.columns else df.iloc[:, 3]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close).astype(float)

        # --- Indicators ---
        bb = BollingerBands(close, window=20, window_dev=2)
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        price = close.iloc[-1]

        bb_pos = ((price - lower) / (upper - lower)) * 2 - 1
        bb_score = -bb_pos
        rsi_val = RSIIndicator(close, window=14).rsi().iloc[-1]
        rsi_score = (50 - rsi_val) / 50

        score = (bb_score + rsi_score) / 2
        return float(max(-1.0, min(1.0, score)))

    except Exception as e:
        print(f"⚠️  meanrev_score failed: {e}")
        return 0.0


if __name__ == "__main__":
    import yfinance as yf

    print("📊 Testing meanrev_score on AAPL...")
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    df = df.rename(columns=str.lower)
    print(f"Mean-reversion score (AAPL): {meanrev_score(df):.3f}")
