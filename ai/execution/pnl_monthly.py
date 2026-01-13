from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class MonthlyCfg:
    tz: str = "America/Chicago"
    base_ccy: str = "USD"
    daily_csv: Path = Path("data/reports/phase36_pnl_daily.csv")
    monthly_csv: Path = Path("data/reports/phase36_monthly_summary.csv")
    chart_png: Path = Path("data/plots/monthly_pnl_cumulative.png")
    lookback_months: int = 12

class MonthlyPnL:
    def __init__(self, cfg: MonthlyCfg):
        self.cfg = cfg

    def _load_daily(self) -> pd.DataFrame:
        p = Path(self.cfg.daily_csv)
        if not p.exists():
            logger.warning("Daily PnL file missing: %s", p)
            return pd.DataFrame(columns=["date", "symbol", "realized_pnl", "fees"])
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in ("date","symbol","realized_pnl","fees"):
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {p}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
        if (df["symbol"]=="__TOTAL__").any():
            day = df[df["symbol"]=="__TOTAL__"][["date","realized_pnl","fees"]].copy()
        else:
            day = df.groupby("date",as_index=False)[["realized_pnl","fees"]].sum()
        return day.sort_values("date").reset_index(drop=True)

    def aggregate_monthly(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        day = self._load_daily()
        if day.empty:
            return day, pd.DataFrame(columns=["month_end","realized_pnl","fees","days"])
        ser = pd.to_datetime(day["date"])
        month = ser.dt.to_period("M").dt.end_time.dt.date
        agg = day.copy()
        agg["month_end"] = month
        msum = agg.groupby("month_end",as_index=False).agg(
            realized_pnl=("realized_pnl","sum"),
            fees=("fees","sum"),
            days=("date","count")
        ).sort_values("month_end")
        if self.cfg.lookback_months>0:
            msum = msum.tail(self.cfg.lookback_months)
        return day, msum

    def save_monthly_csv(self, msum: pd.DataFrame):
        Path(self.cfg.monthly_csv).parent.mkdir(parents=True,exist_ok=True)
        msum.to_csv(self.cfg.monthly_csv,index=False)

    def plot_cumulative_chart(self, msum: pd.DataFrame):
        if msum.empty: 
            logger.warning("No monthly data to plot.")
            return
        Path(self.cfg.chart_png).parent.mkdir(parents=True,exist_ok=True)
        msum["cum_pnl"]=msum["realized_pnl"].cumsum()
        plt.figure(figsize=(8,4))
        plt.plot(msum["month_end"], msum["cum_pnl"], marker="o")
        plt.title(f"Cumulative PnL ({self.cfg.base_ccy})")
        plt.xlabel("Month End")
        plt.ylabel("Cumulative Realized PnL")
        plt.xticks(rotation=45,ha="right")
        plt.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.chart_png,dpi=120)
        plt.close()

    def summarize_month(self, msum: pd.DataFrame)->dict:
        if msum.empty:
            return {"month_end":None,"pnl":0.0,"fees":0.0,"days":0}
        last=msum.iloc[-1]
        return {
            "month_end":str(last["month_end"]),
            "pnl":float(last["realized_pnl"]),
            "fees":float(last["fees"]),
            "days":int(last["days"])
        }
