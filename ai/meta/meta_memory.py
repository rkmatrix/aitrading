# ai/meta/meta_memory.py
from __future__ import annotations
import os, sqlite3, json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

DEFAULT_DB = "data/meta/meta_memory.sqlite3"

@dataclass
class DailySummary:
    date: str                 # 'YYYY-MM-DD' (UTC or market local; be consistent)
    regime: str               # 'bull' | 'bear' | 'sideways'
    pnl: float                # total PnL for the day
    sharpe_like: float        # rough SR proxy for the day
    winrate: float            # 0..1
    trades: int               # number of fills/orders considered
    vol_avg: float            # average volatility signal used
    notes: str = ""           # freeform JSON string for extra metrics

class MetaMemory:
    def __init__(self, db_path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as cx:
            cx.execute("""
            CREATE TABLE IF NOT EXISTS daily (
                date TEXT PRIMARY KEY,
                regime TEXT,
                pnl REAL,
                sharpe_like REAL,
                winrate REAL,
                trades INTEGER,
                vol_avg REAL,
                notes TEXT
            )
            """)
            cx.execute("CREATE INDEX IF NOT EXISTS idx_daily_regime ON daily(regime)")
            cx.commit()

    def add_daily(self, s: DailySummary) -> None:
        d = asdict(s)
        with self._conn() as cx:
            cx.execute("""
            INSERT INTO daily(date, regime, pnl, sharpe_like, winrate, trades, vol_avg, notes)
            VALUES(:date, :regime, :pnl, :sharpe_like, :winrate, :trades, :vol_avg, :notes)
            ON CONFLICT(date) DO UPDATE SET
              regime=excluded.regime,
              pnl=excluded.pnl,
              sharpe_like=excluded.sharpe_like,
              winrate=excluded.winrate,
              trades=excluded.trades,
              vol_avg=excluded.vol_avg,
              notes=excluded.notes
            """, d)
            cx.commit()

    def recent(self, n: int = 30) -> List[DailySummary]:
        with self._conn() as cx:
            rows = cx.execute("SELECT date, regime, pnl, sharpe_like, winrate, trades, vol_avg, notes "
                              "FROM daily ORDER BY date DESC LIMIT ?", (n,)).fetchall()
        out: List[DailySummary] = []
        for r in rows:
            out.append(DailySummary(*r))
        return out

    def stats_by_regime(self, lookback: int = 90) -> Dict[str, Dict[str, float]]:
        with self._conn() as cx:
            rows = cx.execute("""
            SELECT regime,
                   COUNT(*) as days,
                   AVG(pnl) as avg_pnl,
                   AVG(sharpe_like) as avg_sr,
                   AVG(winrate) as avg_wr,
                   AVG(vol_avg) as avg_vol
            FROM daily
            WHERE date >= DATE('now', ?)
            GROUP BY regime
            """, (f"-{lookback} days",)).fetchall()
        stats = {}
        for reg, days, ap, sr, wr, av in rows:
            stats[reg] = {
                "days": float(days or 0),
                "avg_pnl": float(ap or 0),
                "avg_sr": float(sr or 0),
                "avg_wr": float(wr or 0),
                "avg_vol": float(av or 0),
            }
        return stats

    def suggest_blend(self, current_regime: str, lookback: int = 90) -> Dict[str, float]:
        """
        Simple, robust suggestion: weight by positive Sharpe-like and pnl,
        with a bias for current regime. Normalized to sum=1.
        """
        stats = self.stats_by_regime(lookback=lookback)
        regs = ["bull", "bear", "sideways"]
        base = {r: 0.0 for r in regs}
        for r in regs:
            st = stats.get(r, {})
            score = max(0.0, st.get("avg_sr", 0.0)) * 0.6 + max(0.0, st.get("avg_pnl", 0.0)) * 0.4
            base[r] = score
        # regime bias
        if current_regime in base:
            base[current_regime] += 0.25  # gentle nudge
        # normalize
        tot = sum(base.values())
        if tot <= 1e-8:
            return {r: (1.0/3.0) for r in regs}
        return {r: base[r]/tot for r in regs}
