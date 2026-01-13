from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class WeeklyCfg:
    tz: str = "America/Chicago"
    base_ccy: str = "USD"
    daily_csv: Path = Path("data/reports/phase36_pnl_daily.csv")
    weekly_csv: Path = Path("data/reports/phase36_weekly_summary.csv")
    chart_png: Path = Path("data/plots/weekly_pnl.png")
    week_ending: str = "SUN"          # pandas offset suffix
    lookback_weeks: int = 8
    daily_chart_lookback_days: int = 14

class WeeklyPnL:
    def __init__(self, cfg: WeeklyCfg):
        self.cfg = cfg

    def _load_daily(self) -> pd.DataFrame:
        p = Path(self.cfg.daily_csv)
        if not p.exists():
            logger.warning("Daily PnL file missing: %s", p)
            return pd.DataFrame(columns=["date", "symbol", "realized_pnl", "fees"])
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        # required columns
        for c in ("date", "symbol", "realized_pnl", "fees"):
            if c not in df.columns:
                raise ValueError(f"Daily CSV missing column '{c}'")
        # Coerce types
        # Parse as datetime (naive)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Try to localize or convert safely
        try:
            # if already tz-aware -> convert; else localize
            if df["date"].dt.tz is None:
                df["date"] = df["date"].dt.tz_localize(self.cfg.tz, nonexistent="shift_forward", ambiguous="NaT")
            else:
                df["date"] = df["date"].dt.tz_convert(self.cfg.tz)
        except Exception:
            # fallback: if localization fails for mixed/naive values
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(self.cfg.tz)


        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)

        # Focus on per-day totals; prefer __TOTAL__ row when present; else sum symbols
        has_total = (df["symbol"] == "__TOTAL__").any()
        if has_total:
            day = df[df["symbol"] == "__TOTAL__"][["date", "realized_pnl", "fees"]].copy()
        else:
            day = df.groupby("date", as_index=False)[["realized_pnl", "fees"]].sum(numeric_only=True)
        # Normalize date to date (no time)
        day["date"] = day["date"].dt.date
        return day.sort_values("date").reset_index(drop=True)

    def _week_alias(self) -> str:
        # pandas weekly frequency notation, e.g., 'W-SUN' ends on Sunday
        end = (self.cfg.week_ending or "SUN").upper()
        return f"W-{end}"

    def aggregate_weekly(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        day = self._load_daily()
        if day.empty:
            return day, pd.DataFrame(columns=["week_end", "realized_pnl", "fees", "days", "win_days", "loss_days"])

        # Build weekly labels with desired week ending
        ser = pd.to_datetime(day["date"])
        week = ser.dt.to_period(self._week_alias()).dt.end_time.dt.date
        agg = day.copy()
        agg["week_end"] = week

        week_sum = agg.groupby("week_end", as_index=False).agg(
            realized_pnl=("realized_pnl", "sum"),
            fees=("fees", "sum"),
            days=("date", "count"),
            win_days=("realized_pnl", lambda s: int((s > 0).sum())),
            loss_days=("realized_pnl", lambda s: int((s < 0).sum())),
        ).sort_values("week_end")

        # trim lookback
        if self.cfg.lookback_weeks and self.cfg.lookback_weeks > 0:
            week_sum = week_sum.tail(self.cfg.lookback_weeks)

        return day, week_sum

    def save_weekly_csv(self, week_sum: pd.DataFrame) -> None:
        Path(self.cfg.weekly_csv).parent.mkdir(parents=True, exist_ok=True)
        week_sum.to_csv(self.cfg.weekly_csv, index=False)

    def plot_daily_chart(self, day: pd.DataFrame) -> None:
        if day.empty:
            logger.warning("No daily data to chart.")
            return
        if self.cfg.daily_chart_lookback_days and self.cfg.daily_chart_lookback_days > 0:
            day = day.tail(self.cfg.daily_chart_lookback_days)

        dates = pd.to_datetime(day["date"])
        vals = day["realized_pnl"].astype(float)

        Path(self.cfg.chart_png).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.bar(dates, vals)  # default color, single plot, no style
        plt.title("Daily Realized PnL (Last {} days)".format(len(day)))
        plt.xlabel("Date")
        plt.ylabel(f"Realized PnL ({self.cfg.base_ccy})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.cfg.chart_png, dpi=120)
        plt.close()

    def summarize_week(self, week_sum: pd.DataFrame) -> dict:
        if week_sum.empty:
            return {"week_end": None, "pnl": 0.0, "fees": 0.0, "days": 0, "win_days": 0, "loss_days": 0}
        last = week_sum.iloc[-1]
        return {
            "week_end": str(last["week_end"]),
            "pnl": float(last["realized_pnl"]),
            "fees": float(last["fees"]),
            "days": int(last["days"]),
            "win_days": int(last["win_days"]),
            "loss_days": int(last["loss_days"]),
        }
