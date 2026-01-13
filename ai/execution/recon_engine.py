from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import glob

logger = logging.getLogger(__name__)

@dataclass
class ReconConfig:
    tz: str = "America/Chicago"
    local_execs: Path = Path("data/runtime/executions.csv")
    broker_fills_glob: str = "data/broker/fills_*.csv"
    broker_statements_glob: Optional[str] = "data/broker/statements_*.csv"
    out_recon_deltas: Path = Path("data/reports/phase36_recon_deltas.csv")

class Reconciler:
    """
    Reconciles local executions vs broker fills.
    Matching rule (tolerant):
      - By symbol, side, and |qty - qty_broker| <= tol_qty
      - |price - price_broker| <= tol_price_ratio * price
      - |ts_local - ts_broker| <= tol_sec
      - If multiple candidates, pick argmin of score (timestamp diff, then price diff).
    Emits 'missing_local', 'missing_broker', 'mismatch' rows.
    """
    def __init__(self, cfg: ReconConfig,
                 tol_qty: float = 1e-6,
                 tol_price_ratio: float = 0.005,  # 0.5%
                 tol_sec: int = 120):
        self.cfg = cfg
        self.tol_qty = tol_qty
        self.tol_price_ratio = tol_price_ratio
        self.tol_sec = tol_sec

    # ---- helpers ----
    def _read_execs(self, p: Path) -> pd.DataFrame:
        req = ("ts", "symbol", "side", "qty", "price")
        if not Path(p).exists():
            logger.warning("Local executions file missing: %s", p)
            return pd.DataFrame(columns=req)
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {p}")
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["side"] = df["side"].str.upper()
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        if "fee" not in df.columns:
            df["fee"] = 0.0
        else:
            df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
        if "exec_id" not in df.columns:
            df["exec_id"] = pd.NA
        if "order_id" not in df.columns:
            df["order_id"] = pd.NA
        return df.sort_values("ts").reset_index(drop=True)

    def _read_broker_fills(self, pattern: str) -> pd.DataFrame:
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning("No broker fills found for pattern: %s", pattern)
            return pd.DataFrame(columns=["ts", "symbol", "side", "qty", "price", "fee", "exec_id", "order_id", "source"])
        frames = []
        for fp in files:
            df = pd.read_csv(fp)
            df.columns = [c.strip().lower() for c in df.columns]
            # tolerant header mapping
            m = {
                "timestamp": "ts",
                "time": "ts",
                "quantity": "qty",
                "commission": "fee",
            }
            for k, v in m.items():
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            need = ("ts", "symbol", "side", "qty", "price")
            for c in need:
                if c not in df.columns:
                    raise ValueError(f"Broker fills '{fp}' missing column '{c}'")
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df["symbol"] = df["symbol"].astype(str).str.upper()
            df["side"] = df["side"].str.upper()
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            if "fee" not in df.columns:
                df["fee"] = 0.0
            else:
                df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
            if "exec_id" not in df.columns:
                df["exec_id"] = pd.NA
            if "order_id" not in df.columns:
                df["order_id"] = pd.NA
            df["source"] = Path(fp).name
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("ts").reset_index(drop=True)

    # ---- recon ----
    def reconcile(self) -> pd.DataFrame:
        local = self._read_execs(self.cfg.local_execs)
        broker = self._read_broker_fills(self.cfg.broker_fills_glob)

        if local.empty and broker.empty:
            logger.warning("No data to reconcile.")
            return pd.DataFrame(columns=["kind", "symbol", "side", "qty_local", "qty_broker",
                                         "price_local", "price_broker", "fee_local", "fee_broker",
                                         "ts_local", "ts_broker", "order_id_local", "order_id_broker",
                                         "exec_id_local", "exec_id_broker", "src_broker"])

        used_broker = np.zeros(len(broker), dtype=bool)
        deltas: List[Dict] = []

        for i, lr in local.iterrows():
            sym, side, lq, lp, lf, lts = lr["symbol"], lr["side"], float(lr["qty"]), float(lr["price"]), float(lr["fee"]), lr["ts"]
            # candidate broker rows: same sym/side, within time window
            mask = (
                (broker["symbol"] == sym) &
                (broker["side"] == side) &
                (~used_broker) &
                (broker["ts"].between(lts - pd.Timedelta(seconds=self.tol_sec),
                                      lts + pd.Timedelta(seconds=self.tol_sec)))
            )
            cands = broker[mask].copy()
            if not cands.empty:
                # score: ts diff then price diff then qty diff
                cands["score"] = (
                    (cands["ts"] - lts).abs() / pd.Timedelta(seconds=1)
                ).astype(float) + (abs(cands["price"] - lp) / max(lp, 1e-9)) * 10.0 + abs(cands["qty"] - lq)
                k = int(cands["score"].idxmin())
                br = broker.loc[k]
                # check tolerances
                q_ok = abs(br["qty"] - lq) <= self.tol_qty
                p_ok = abs(br["price"] - lp) <= self.tol_price_ratio * max(lp, 1e-9)
                if q_ok and p_ok:
                    used_broker[k] = True
                    # fee diff tolerance: allow small cents drift
                    fee_diff = float(br["fee"]) - lf
                    if abs(fee_diff) > 0.02:
                        deltas.append({
                            "kind": "mismatch_fee",
                            "symbol": sym, "side": side,
                            "qty_local": lq, "qty_broker": float(br["qty"]),
                            "price_local": lp, "price_broker": float(br["price"]),
                            "fee_local": lf, "fee_broker": float(br["fee"]),
                            "ts_local": lts, "ts_broker": br["ts"],
                            "order_id_local": lr.get("order_id", pd.NA),
                            "order_id_broker": br.get("order_id", pd.NA),
                            "exec_id_local": lr.get("exec_id", pd.NA),
                            "exec_id_broker": br.get("exec_id", pd.NA),
                            "src_broker": br.get("source", pd.NA)
                        })
                else:
                    deltas.append({
                        "kind": "mismatch",
                        "symbol": sym, "side": side,
                        "qty_local": lq, "qty_broker": float(br["qty"]),
                        "price_local": lp, "price_broker": float(br["price"]),
                        "fee_local": lf, "fee_broker": float(br["fee"]),
                        "ts_local": lts, "ts_broker": br["ts"],
                        "order_id_local": lr.get("order_id", pd.NA),
                        "order_id_broker": br.get("order_id", pd.NA),
                        "exec_id_local": lr.get("exec_id", pd.NA),
                        "exec_id_broker": br.get("exec_id", pd.NA),
                        "src_broker": br.get("source", pd.NA)
                    })
            else:
                deltas.append({
                    "kind": "missing_broker",
                    "symbol": sym, "side": side,
                    "qty_local": lq, "qty_broker": np.nan,
                    "price_local": lp, "price_broker": np.nan,
                    "fee_local": lf, "fee_broker": np.nan,
                    "ts_local": lts, "ts_broker": pd.NaT,
                    "order_id_local": lr.get("order_id", pd.NA),
                    "order_id_broker": pd.NA,
                    "exec_id_local": lr.get("exec_id", pd.NA),
                    "exec_id_broker": pd.NA,
                    "src_broker": pd.NA
                })

        # broker rows not used => missing local
        for k, br in broker[~used_broker].iterrows():
            deltas.append({
                "kind": "missing_local",
                "symbol": br["symbol"], "side": br["side"],
                "qty_local": np.nan, "qty_broker": float(br["qty"]),
                "price_local": np.nan, "price_broker": float(br["price"]),
                "fee_local": np.nan, "fee_broker": float(br["fee"]),
                "ts_local": pd.NaT, "ts_broker": br["ts"],
                "order_id_local": pd.NA, "order_id_broker": br.get("order_id", pd.NA),
                "exec_id_local": pd.NA, "exec_id_broker": br.get("exec_id", pd.NA),
                "src_broker": br.get("source", pd.NA)
            })

        out = pd.DataFrame(deltas)
        if not out.empty:
            out["date_local"] = pd.to_datetime(out["ts_local"], utc=True, errors="coerce").dt.tz_convert(self.cfg.tz).dt.date
            out["date_broker"] = pd.to_datetime(out["ts_broker"], utc=True, errors="coerce").dt.tz_convert(self.cfg.tz).dt.date

        # write report
        self.cfg.out_recon_deltas.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.cfg.out_recon_deltas, index=False)
        return out
