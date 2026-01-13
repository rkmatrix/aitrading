from __future__ import annotations
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EquityCfg:
    tz: str = "America/Chicago"
    base_ccy: str = "USD"
    daily_pnl_csv: Path = Path("data/reports/phase36_pnl_daily.csv")
    broker_equity_glob: str = "data/broker/equity_*.csv"
    drift_csv: Path = Path("data/reports/phase37_equity_drift.csv")
    max_abs_drift_pct: float = 0.5
    min_equity_usd: float = 1000.0


class EquityReconciler:
    def __init__(self, cfg: EquityCfg):
        self.cfg = cfg

    # ----------------------------------------------------------------------
    def _load_daily_pnl(self) -> pd.DataFrame:
        """Load bot-side daily realized PnL file (Phase 36 output)."""
        p = Path(self.cfg.daily_pnl_csv)
        if not p.exists():
            logger.warning("Missing daily PnL file: %s", p)
            return pd.DataFrame(columns=["date", "realized_pnl"])

        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]

        if "realized_pnl" not in df.columns:
            raise ValueError("Missing 'realized_pnl' column in daily CSV")

        # prefer total rows if present
        if "symbol" in df.columns and (df["symbol"] == "__TOTAL__").any():
            df = df[df["symbol"] == "__TOTAL__"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        df = df.groupby("date", as_index=False)["realized_pnl"].sum()
        return df.sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------------------
    def _load_broker_equity(self) -> pd.DataFrame:
        """Load broker-side equity snapshots from equity_*.csv files."""
        files = sorted(glob.glob(self.cfg.broker_equity_glob))
        if not files:
            logger.warning("No broker equity files found: %s", self.cfg.broker_equity_glob)
            return pd.DataFrame(columns=["date", "broker_equity"])

        frames = []
        for fp in files:
            df = pd.read_csv(fp)
            df.columns = [c.strip().lower() for c in df.columns]
            if "equity" not in df.columns:
                raise ValueError(f"{fp} missing 'equity' column")
            if "date" not in df.columns:
                if "ts" in df.columns:
                    df["date"] = pd.to_datetime(df["ts"], errors="coerce")
                else:
                    raise ValueError(f"{fp} missing 'date' or 'ts' column")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df["broker_equity"] = pd.to_numeric(df["equity"], errors="coerce").fillna(0.0)
            df["source"] = Path(fp).name
            frames.append(df[["date", "broker_equity", "source"]])

        all_df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        return all_df

    # ----------------------------------------------------------------------
    def reconcile(self) -> pd.DataFrame:
        """Compare internal vs broker equity and compute drift %."""
        pnl = self._load_daily_pnl()
        brok = self._load_broker_equity()

        if pnl.empty or brok.empty:
            logger.warning("Insufficient data for equity reconciliation.")
            return pd.DataFrame(
                columns=["date", "internal_equity", "broker_equity", "drift_pct", "alert"]
            )

        # ensure datetime dtype for merge_asof
        pnl["date"] = pd.to_datetime(pnl["date"], errors="coerce")
        brok["date"] = pd.to_datetime(brok["date"], errors="coerce")

        # compute internal cumulative equity (start from first broker value)
        start_equity = float(brok.iloc[0]["broker_equity"])
        pnl["cum_pnl"] = pnl["realized_pnl"].cumsum()
        pnl["internal_equity"] = start_equity + pnl["cum_pnl"]

        # sort both for merge_asof
        pnl = pnl.sort_values("date").reset_index(drop=True)
        brok = brok.sort_values("date").reset_index(drop=True)

        # perform merge
        merged = pd.merge_asof(
            brok,
            pnl,
            on="date",
            direction="backward"
        )

        # fill any NaNs
        merged["broker_equity"] = pd.to_numeric(merged["broker_equity"], errors="coerce").fillna(0.0)
        merged["internal_equity"] = pd.to_numeric(merged["internal_equity"], errors="coerce").fillna(0.0)

        # compute drift %
        with np.errstate(divide="ignore", invalid="ignore"):
            merged["drift_pct"] = 100.0 * (
                (merged["broker_equity"] - merged["internal_equity"])
                / merged["internal_equity"].replace(0, np.nan)
            )

        # assign alert status
        merged["alert"] = "OK"
        merged.loc[
            merged["drift_pct"].abs() > self.cfg.max_abs_drift_pct, "alert"
        ] = "DRIFT_EXCEEDED"
        merged.loc[
            merged["broker_equity"] < self.cfg.min_equity_usd, "alert"
        ] = "LOW_EQUITY"

        # write CSV
        Path(self.cfg.drift_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.cfg.drift_csv, index=False)

        logger.info(
            "✅ Reconciliation complete: %d rows -> %s",
            len(merged),
            self.cfg.drift_csv
        )
        return merged
