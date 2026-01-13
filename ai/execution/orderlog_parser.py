from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional

TZ = "America/New_York"  # market tz; change if needed

# Column aliasing to normalize different broker dumps
ALIASES = {
    "timestamp": ["timestamp", "ts", "submitted_at", "created_at", "time"],
    "symbol": ["symbol", "sym", "ticker"],
    "side": ["side", "action"],
    "qty": ["qty", "quantity", "filled_qty", "filled_qty_total"],
    "order_type": ["order_type", "type"],
    "limit_price": ["limit_price", "limit", "price"],
    "fill_price": ["fill_price", "avg_fill_price", "filled_avg_price", "average_price"],
    "status": ["status", "state"],
    "fees": ["fees", "commission", "commissions"],
    "slippage_bps": ["slippage_bps", "slippage"],
    "latency_ms": ["latency_ms", "latency", "lat_ms"],
    "best_bid": ["best_bid", "bid"],
    "best_ask": ["best_ask", "ask"],
    "mid": ["mid", "mid_price"],
    "spread": ["spread", "spread_abs"],
    "exchange": ["exchange", "venue"],
    "strategy_id": ["strategy_id", "strategy", "algo"],
    "order_id": ["order_id", "id"],
    "parent_id": ["parent_id", "client_order_id", "client_id"],
    "tags": ["tags", "tag"],
    "cash_before": ["cash_before"],
    "cash_after": ["cash_after"]
}

STATUS_NORMAL = {
    "filled": "filled",
    "partially_filled": "partial",
    "partial": "partial",
    "canceled": "canceled",
    "cancelled": "canceled",
    "rejected": "rejected",
    "new": "new",
    "accepted": "new",
    "open": "open",
    "done": "filled"
}

SIDE_NORMAL = {"buy": "buy", "sell": "sell", "sell_short": "sell"}


def _pick(colnames: List[str], aliases: List[str]) -> Optional[str]:
    s = set(c.lower() for c in colnames)
    for a in aliases:
        if a.lower() in s:
            return a
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    cols = list(df.columns)
    for canon, aliases in ALIASES.items():
        pick = _pick(cols, aliases)
        if pick:
            mapping[pick] = canon
    out = df.rename(columns=mapping)

    # derive mid/spread if needed
    if "mid" not in out and {"best_bid", "best_ask"}.issubset(out.columns):
        out["mid"] = (out["best_bid"].astype(float) + out["best_ask"].astype(float)) / 2.0
    if "spread" not in out and {"best_bid", "best_ask"}.issubset(out.columns):
        out["spread"] = (out["best_ask"].astype(float) - out["best_bid"].astype(float)).clip(lower=0)

    # normalize side/status
    if "side" in out:
        out["side"] = out["side"].astype(str).str.lower().map(lambda x: SIDE_NORMAL.get(x, x))
    if "status" in out:
        out["status"] = out["status"].astype(str).str.lower().map(lambda x: STATUS_NORMAL.get(x, x))
    return out


def parse_logs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        if "timestamp" in df:
            # robust tz handling: localize if naive; convert to NY then to UTC
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            if ts.dt.tz is not None:
                df["timestamp"] = ts.dt.tz_convert(TZ).dt.tz_convert("UTC")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(
                    TZ, nonexistent="shift_forward", ambiguous="NaT"
                ).dt.tz_convert("UTC")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full.sort_values(["symbol", "timestamp"], inplace=True)

    # numeric coercions
    for c in [
        "qty", "limit_price", "fill_price", "fees", "slippage_bps", "latency_ms",
        "best_bid", "best_ask", "mid", "spread", "cash_before", "cash_after"
    ]:
        if c in full:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    return full
