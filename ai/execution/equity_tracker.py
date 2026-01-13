from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import logging
import datetime as dt

logger = logging.getLogger(__name__)

@dataclass
class TrackerCfg:
    tz: str = "America/Chicago"
    base_ccy: str = "USD"
    drift_csv: Path = Path("data/reports/phase37_equity_drift.csv")
    history_csv: Path = Path("data/reports/phase37_equity_history.csv")
    chart_png: Path = Path("data/plots/equity_drift_chart.png")
    lookback_days: int = 60
    alert_drift_threshold: float = 0.5
    tag: str = "phase37_equity_tracker"


class EquityTracker:
    def __init__(self, cfg: TrackerCfg):
        self.cfg = cfg

    # ----------------------------------------------------------------------
    def _load_latest_drift(self) -> pd.DataFrame:
        """Load most recent drift reconciliation output."""
        p = self.cfg.drift_csv
        if not p.exists():
            logger.warning("Drift file missing: %s", p)
            return pd.DataFrame(columns=["date", "internal_equity", "broker_equity", "drift_pct"])
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in ("date", "internal_equity", "broker_equity", "drift_pct"):
            if c not in df.columns:
                raise ValueError(f"Drift CSV missing column: {c}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        return df.sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------------------
    def update_history(self) -> pd.DataFrame:
        """Append latest drift data into persistent history file."""
        latest = self._load_latest_drift()
        if latest.empty:
            logger.warning("No new drift data to append.")
            return latest

        hist_path = self.cfg.history_csv
        hist_path.parent.mkdir(parents=True, exist_ok=True)

        if hist_path.exists() and hist_path.stat().st_size > 0:
            hist = pd.read_csv(hist_path)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=False)
        else:
            hist = pd.DataFrame(columns=["date", "internal_equity", "broker_equity", "drift_pct"])

        # Avoid FutureWarning: filter out empty frames
        pieces = [h for h in [hist, latest] if not h.empty]
        merged = pd.concat(pieces, ignore_index=True)
        merged.drop_duplicates(subset=["date"], keep="last", inplace=True)
        merged.sort_values("date", inplace=True)
        merged.to_csv(hist_path, index=False)

        logger.info("📈 History updated → %s (%d rows)", hist_path, len(merged))
        return merged

    # ----------------------------------------------------------------------
    def plot_chart(self, hist: pd.DataFrame) -> None:
        """Plot broker vs internal equity comparison line chart."""
        if hist.empty:
            logger.warning("No history to plot.")
            return

        # Normalize to timezone-naive datetime64[ns]
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.tz_localize(None)

        # Apply lookback safely
        if self.cfg.lookback_days > 0:
            cut = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=self.cfg.lookback_days)
            hist = hist[hist["date"] >= cut]

        if hist.empty:
            logger.warning("No history left after applying lookback.")
            return

        Path(self.cfg.chart_png).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(hist["date"], hist["internal_equity"], label="Internal", marker="o")
        plt.plot(hist["date"], hist["broker_equity"], label="Broker", marker="x")
        plt.title(f"Equity Tracker – {self.cfg.base_ccy}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.chart_png, dpi=120)
        plt.close()

        logger.info("🖼️ Chart saved → %s", self.cfg.chart_png)

    # ----------------------------------------------------------------------
    def summarize_last(self, hist: pd.DataFrame) -> dict:
        """Return latest snapshot summary and drift alert flag."""
        if hist.empty:
            return {"date": None, "drift": 0.0, "alert": False}
        last = hist.iloc[-1]
        alert = abs(float(last["drift_pct"])) > self.cfg.alert_drift_threshold
        return {
            "date": str(last["date"]),
            "internal": float(last["internal_equity"]),
            "broker": float(last["broker_equity"]),
            "drift": float(last["drift_pct"]),
            "alert": alert,
        }
