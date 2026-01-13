from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PnLConfig:
    base_ccy: str = "USD"
    tz: str = "America/Chicago"
    executions_csv: Path = Path("data/runtime/executions.csv")
    positions_csv: Optional[Path] = Path("data/runtime/positions_snapshot.csv")
    prices_root: Optional[Path] = Path("data/prices")
    out_daily_pnl: Path = Path("data/reports/phase36_pnl_daily.csv")
    out_positions: Path = Path("data/reports/phase36_pnl_positions.csv")

class PnLLedger:
    """
    CSV-backed ledger that computes realized/unrealized PnL and fees by symbol and day.
    Assumes executions CSV schema (wide-tolerant):
      ts, symbol, side(BUY/SELL), qty, price, fee(optional), order_id(optional), exec_id(optional)
    """
    def __init__(self, cfg: PnLConfig):
        self.cfg = cfg

    # ---------- Loading ----------
    def _load_csv_if_exists(self, path: Path, required_cols: Tuple[str, ...]) -> Optional[pd.DataFrame]:
        path = Path(path)
        if not path.exists():
            logger.warning("Missing CSV: %s", path)
            return None
        df = pd.read_csv(path)
        # Normalize header case/whitespace
        df.columns = [c.strip().lower() for c in df.columns]
        # Ensure required cols exist
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {path}")
        return df

    def load_executions(self) -> pd.DataFrame:
        req = ("ts", "symbol", "side", "qty", "price")
        df = self._load_csv_if_exists(self.cfg.executions_csv, req)
        if df is None:
            # create fully-typed empty frame
            df = pd.DataFrame(columns=req)
        # ensure proper dtypes
        df["symbol"] = df.get("symbol", "").astype(str).str.upper()
        df["side"] = df.get("side", "").astype(str).str.upper()
        df["qty"] = pd.to_numeric(df.get("qty", 0), errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0.0)
        df["ts"] = pd.to_datetime(df.get("ts", pd.NaT), utc=True, errors="coerce")
        if "fee" not in df.columns:
            df["fee"] = 0.0
        else:
            df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
    
        df = df[(df["qty"] != 0) & (df["price"] > 0)]
        return df.sort_values("ts").reset_index(drop=True)


    def load_positions_snapshot(self) -> Optional[pd.DataFrame]:
        if self.cfg.positions_csv is None:
            return None
        req = ("symbol", "qty", "avg_price")
        df = self._load_csv_if_exists(self.cfg.positions_csv, req)
        if df is None:
            return None
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
        df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0.0)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        return df

    # ---------- Price helpers ----------
    def _load_last_price(self, symbol: str) -> Optional[float]:
        if not self.cfg.prices_root:
            return None
        p = Path(self.cfg.prices_root) / f"{symbol.upper()}.csv"
        if not p.exists():
            return None
        try:
            d = pd.read_csv(p)
            d.columns = [c.strip().lower() for c in d.columns]
            if "close" not in d.columns:
                return None
            if "ts" in d.columns:
                d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
                d = d.sort_values("ts")
            # use last non-null close
            val = pd.to_numeric(d["close"], errors="coerce").dropna()
            return float(val.iloc[-1]) if len(val) else None
        except Exception as e:
            logger.exception("Failed to read price file %s: %s", p, e)
            return None

    # ---------- Core PnL ----------
    def compute_realized_pnl(self, execs: pd.DataFrame) -> pd.DataFrame:
        """
        FIFO realized PnL per trade, aggregated by day & symbol.
        """
        rows = []
        by_sym = dict(tuple(execs.groupby("symbol", sort=False)))
        for sym, df in by_sym.items():
            # maintain FIFO inventory: list of (qty_remaining, cost_basis_per_share)
            fifo: list[tuple[float, float]] = []
            df = df.copy()
            # normalize sign: buys positive qty, sells negative qty
            side_sign = np.where(df["side"] == "BUY", 1.0, -1.0)
            df["signed_qty"] = df["qty"].values * side_sign
            df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)

            for _, r in df.iterrows():
                q = float(r["signed_qty"])
                px = float(r["price"])
                fee = float(r["fee"])
                ts = r["ts"]

                if q > 0:  # BUY -> add to inventory
                    fifo.append((q, px))
                else:      # SELL -> realize vs FIFO
                    sell_qty = -q
                    remain_to_match = sell_qty
                    realized = 0.0
                    while remain_to_match > 1e-9 and fifo:
                        inv_qty, inv_px = fifo[0]
                        take = min(inv_qty, remain_to_match)
                        realized += (px - inv_px) * take
                        inv_qty -= take
                        remain_to_match -= take
                        if inv_qty <= 1e-9:
                            fifo.pop(0)
                        else:
                            fifo[0] = (inv_qty, inv_px)

                    # If selling more than inventory, treat extra as short opening (no realized until closed)
                    # We'll push negative inventory (short) with current price as basis
                    if remain_to_match > 1e-9:
                        fifo.insert(0, (-remain_to_match, px))  # short position

                    rows.append({
                        "ts": ts,
                        "symbol": sym,
                        "trade_pnl": realized - fee,  # include fee on trade line
                        "fee": fee,
                    })

        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "realized_pnl", "fees"])

        out = pd.DataFrame(rows)
        out["date"] = out["ts"].dt.tz_convert(self.cfg.tz).dt.date
        g = out.groupby(["date", "symbol"], as_index=False).agg(
            realized_pnl=("trade_pnl", "sum"),
            fees=("fee", "sum"),
        )
        return g

    def compute_unrealized_pnl(self, execs: pd.DataFrame, pos_snapshot: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Rebuild current positions from executions (fallback), or use provided snapshot if available.
        Mark-to-market using price cache or last trade price fallback.
        """
        # Build positions from execs (FIFO avg)
        ex = execs.copy()
        side_sign = np.where(ex["side"] == "BUY", 1.0, -1.0)
        ex["signed_qty"] = ex["qty"].values * side_sign

        pos = ex.groupby("symbol", as_index=False).agg(
            qty=("signed_qty", "sum")
        )
        # compute avg cost from remaining inventory approximation using VWAP of net buys vs sells
        # Simpler robust approach: maintain running position with avg cost
        avg_rows = []
        for sym, df in ex.groupby("symbol"):
            qty = 0.0
            avg_cost = 0.0
            for _, r in df.iterrows():
                q = float(r["signed_qty"])
                px = float(r["price"])
                if q > 0:  # buy -> average up/down
                    new_qty = qty + q
                    if abs(new_qty) < 1e-9:
                        qty = 0.0
                        avg_cost = 0.0
                    elif qty >= 0:
                        avg_cost = (qty * avg_cost + q * px) / new_qty
                        qty = new_qty
                    else:
                        # covering short
                        qty = new_qty
                        if qty > 0:
                            avg_cost = px  # crossed through zero; set basis to trade price
                else:  # sell / short more
                    new_qty = qty + q
                    if qty <= 0 and q < 0:
                        # adding to short -> average
                        new_qty_abs = abs(new_qty)
                        qty_abs = abs(qty)
                        q_abs = abs(q)
                        if new_qty_abs > 1e-9:
                            avg_cost = (qty_abs * avg_cost + q_abs * px) / new_qty_abs
                    else:
                        # reducing long or covering short -> avg_cost unchanged unless cross zero
                        if np.sign(qty) != np.sign(new_qty) and abs(new_qty) > 1e-9:
                            avg_cost = px
                    qty = new_qty
            avg_rows.append({"symbol": sym, "qty": qty, "avg_price": avg_cost})

        built_pos = pd.DataFrame(avg_rows)
        # If positions snapshot exists, prefer it
        if pos_snapshot is not None and not pos_snapshot.empty:
            # unify symbols casing
            pos_snapshot = pos_snapshot.copy()
            pos_snapshot["symbol"] = pos_snapshot["symbol"].str.upper()
            # merge preferring snapshot qty/avg
            merged = pd.merge(built_pos, pos_snapshot[["symbol", "qty", "avg_price"]],
                              on="symbol", how="outer", suffixes=("_built", ""))
            merged["qty"] = merged["qty"].fillna(merged["qty_built"]).fillna(0.0)
            merged["avg_price"] = merged["avg_price"].fillna(merged["avg_price_built"]).fillna(0.0)
            positions = merged[["symbol", "qty", "avg_price"]].copy()
        else:
            positions = built_pos

        # Price each symbol
        marks = []
        for _, r in positions.iterrows():
            sym = r["symbol"]; qty = float(r["qty"]); avg_px = float(r["avg_price"])
            if abs(qty) < 1e-9:
                continue
            last_px = self._load_last_price(sym)
            if last_px is None:
                # fallback to last execution price for symbol
                last_exec_px = ex.loc[ex["symbol"] == sym, "price"]
                last_px = float(last_exec_px.iloc[-1]) if len(last_exec_px) else avg_px
            unreal = (last_px - avg_px) * qty
            marks.append({
                "symbol": sym, "qty": qty, "avg_price": avg_px,
                "mark": last_px, "unrealized_pnl": unreal
            })
        return pd.DataFrame(marks)

    # ---------- Orchestration ----------
    def run(self) -> Dict[str, pd.DataFrame]:
        execs = self.load_executions()
        pos_snapshot = self.load_positions_snapshot()

        realized = self.compute_realized_pnl(execs)
        unreal = self.compute_unrealized_pnl(execs, pos_snapshot)

                # daily table
        daily = realized.copy()
        # Add per-day total across symbols
        if not daily.empty:
            totals = (
                daily.groupby("date", as_index=False)[["realized_pnl", "fees"]]
                .sum(numeric_only=True)
            )
            totals["symbol"] = "__TOTAL__"
            daily = pd.concat(
                [daily, totals[["date", "symbol", "realized_pnl", "fees"]]],
                ignore_index=True,
            )


        # persist
        self.cfg.out_daily_pnl.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.out_positions.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(self.cfg.out_daily_pnl, index=False)
        (unreal if unreal is not None else pd.DataFrame()).to_csv(self.cfg.out_positions, index=False)

        return {"executions": execs, "daily": daily, "positions": unreal}
