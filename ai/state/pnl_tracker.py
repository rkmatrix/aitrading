from __future__ import annotations
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DailyPnlProvider:
    csv_path: Optional[str] = None
    tz: str = "America/Chicago"
    fallback_env_key: str = "PHASE24_FAKE_TODAY_PNL"

    def today_pnl(self) -> float:
        """
        Returns today's realized PnL (float).
        Priority:
        1) csv_path if exists with rows like: date,pnl  (date = YYYY-MM-DD)
        2) env var fallback (used by smoke test)
        3) 0.0 if nothing else is available
        """
        # 1) CSV
        if self.csv_path:
            p = Path(self.csv_path)
            if p.exists():
                today = dt.date.today().isoformat()
                try:
                    with p.open(newline="", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        last = None
                        for row in r:
                            if row.get("date") == today:
                                return float(row.get("pnl") or 0.0)
                            last = row
                        # If no today's row, return 0.0 (conservative)
                        return 0.0
                except Exception:
                    pass

        # 2) env fallback
        import os
        if self.fallback_env_key in os.environ:
            try:
                return float(os.environ[self.fallback_env_key])
            except Exception:
                return 0.0

        # 3) default
        return 0.0
