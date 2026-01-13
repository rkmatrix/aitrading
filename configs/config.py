# Risk Defaults (you can keep using CLI flags to override)
PHASE7_RISK = {
    "MAX_DD": 0.15,
    "VOLA_LOOKBACK": 30,
    "VOLA_CAP": 0.04,
    "MARKET_OPEN": "09:30",
    "MARKET_CLOSE": "16:00",
    "FORCE_FLAT_OUTSIDE": True,
    "MIN_EQUITY": 1000.0,
    "RISK_PER_TRADE": 0.01,
}

PORTFOLIO = {
    "max_gross_exposure": 0.95,     # 95% invested max
    "max_net_exposure": 0.95,       # long-only default; keep for future shorting
    "cash_buffer": 0.05,            # hold 5% cash
    "max_weight_per_asset": 0.20,   # 20% cap
    "min_weight_abs": 0.01,         # ignore micro-allocations
    "lookback_days": 60,            # corr/vol features
    "regime_clusters": 4,           # if sklearn available
    "vol_kill_switch": 0.08,        # 8% 20d realized vol on SPY triggers de-risk
    "strategy_blends": {            # regime -> weights across (ppo, mom, meanrev)
        "bull_lowvol":  {"ppo": 0.60, "mom": 0.35, "mr": 0.05},
        "bull_highvol": {"ppo": 0.50, "mom": 0.30, "mr": 0.20},
        "bear_lowvol":  {"ppo": 0.35, "mom": 0.15, "mr": 0.50},
        "bear_highvol": {"ppo": 0.25, "mom": 0.10, "mr": 0.65}
    },
    "benchmark_symbol": "SPY"       # for regime & vol reference
}


# Data providers (optional)
POLYGON_API_KEY = "xJp_WXScxCrxHASORcDiqwt54I9Op3lR"            # optional
TWELVEDATA_API_KEY = "41571ec6b3104ef39584de179318e397"         # optional
DATA_PROVIDER_PRIORITY = ["polygon", "twelvedata", "yahoo"]

# Alpaca trading
ALPACA_API_KEY = "PK65UCB4ORXVZNN64QRI43S2HG"
ALPACA_SECRET_KEY = "GtUapPraHsfodj2oKaeUFCgaJawG84ppP1oxDEBhBi8m"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
DEFAULT_ORDER_TIME_IN_FORCE = "day"   # "day" | "gtc" | "opg" | "cls" | "ioc" | "fok"

# Logging backends (optional; enable via CLI flags)
WANDB_ENABLED = False
MLFLOW_ENABLED = False
