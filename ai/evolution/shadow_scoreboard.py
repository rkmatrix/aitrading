"""
Phase 99 – Shadow Scoreboard & Policy Evolution Engine

Evaluates:
    • Primary bot (phase26)
    • Shadow ML bot  (phase26_shadow_ml)
    • Shadow Alt bot (phase26_shadow_alt)

Data sources:
    data/runtime/cluster/shadow_ml_decisions.jsonl
    data/runtime/cluster/shadow_alt_decisions.jsonl
    data/reports/phase37_equity_history.csv     (primary equity)
    data/reports/shadow_pnl_shadow_ml.csv       (optional synthetic pnl)
    data/reports/shadow_pnl_shadow_alt.csv

Outputs:
    data/reports/phase99_scoreboard.json
    data/reports/phase99_leaderboard.csv

Metrics:
    - Cumulative return
    - Daily return stability
    - Drawdown
    - Sharpe-like score
    - "Evolution score" (weighted blend)

Alerts:
    • Telegram notifications when shadow bots outperform primary

Promotion Candidates:
    • If score > threshold → promotion candidate logged
"""

from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify

ensure_env_loaded()
log = logging.getLogger("ShadowScoreboard")


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
@dataclass
class ScoreboardConfig:
    ml_decisions_path: str
    alt_decisions_path: str
    primary_equity_path: str

    min_samples: int = 30
    sharpe_weight: float = 0.5
    dd_weight: float = 0.3
    trend_weight: float = 0.2

    alert_enabled: bool = True
    alert_tag: str = "Phase99Scoreboard"

    promotion_threshold: float = 0.65


# -----------------------------------------------------------
# Utility: load JSONL
# -----------------------------------------------------------
def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# -----------------------------------------------------------
# Compute synthetic PnL (for shadow bots)
# -----------------------------------------------------------
def compute_shadow_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple synthetic PnL model:
    - LONG size * 0.2% per tick
    - SHORT size * 0.2% per tick * -1
    - FLAT = 0
    """
    if df.empty:
        return df

    df = df.copy()
    df["pnl"] = 0.0

    for i, row in df.iterrows():
        side = row.get("side")
        size = float(row.get("size", 0))
        price = float(row.get("price", 0))

        ret = 0.0
        if side == "LONG":
            ret = price * 0.002 * size
        elif side == "SHORT":
            ret = -price * 0.002 * size

        df.at[i, "pnl"] = ret

    df["cum_pnl"] = df["pnl"].cumsum()
    return df


# -----------------------------------------------------------
# Compute performance stats
# -----------------------------------------------------------
def compute_stats(df: pd.DataFrame, pnl_col="pnl") -> Dict[str, float]:
    if df.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "trend_score": 0.0,
        }

    returns = df[pnl_col].values
    if len(returns) < 3:
        return {
            "cumulative_return": float(np.sum(returns)),
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "trend_score": 0.0,
        }

    cum = float(np.sum(returns))
    std = float(np.std(returns) + 1e-6)
    sharpe = float(np.mean(returns) / std)

    equity_curve = np.cumsum(returns)
    peak = np.maximum.accumulate(equity_curve)
    dd = float(np.max(peak - equity_curve) / (np.max(peak) + 1e-9))

    trend = float(np.mean(np.diff(equity_curve)))

    return {
        "cumulative_return": cum,
        "sharpe_like": sharpe,
        "max_drawdown": dd,
        "trend_score": trend,
    }


# -----------------------------------------------------------
# Evolution score (main ranking formula)
# -----------------------------------------------------------
def compute_evolution_score(stats: Dict[str, float], cfg: ScoreboardConfig) -> float:
    # Reduce impact of large DD
    dd_penalty = (1.0 - stats["max_drawdown"])

    score = (
        cfg.sharpe_weight * stats["sharpe_like"]
        + cfg.dd_weight * dd_penalty
        + cfg.trend_weight * stats["trend_score"]
    )

    return float(score)


# -----------------------------------------------------------
# Scoreboard Engine
# -----------------------------------------------------------
class ShadowScoreboard:
    def __init__(self, cfg: ScoreboardConfig):
        self.cfg = cfg

        self.ml_path = Path(cfg.ml_decisions_path)
        self.alt_path = Path(cfg.alt_decisions_path)
        self.primary_path = Path(cfg.primary_equity_path)

        self.out_json = Path("data/reports/phase99_scoreboard.json")
        self.out_csv = Path("data/reports/phase99_leaderboard.csv")

    def run(self):
        log.info("Phase 99 Scoreboard – Loading data...")

        ml_df = load_jsonl(self.ml_path)
        alt_df = load_jsonl(self.alt_path)

        # primary returns from equity history
        primary_df = pd.DataFrame()
        if self.primary_path.exists():
            primary_df = pd.read_csv(self.primary_path)
            if "equity" in primary_df.columns:
                primary_df["pnl"] = primary_df["equity"].diff().fillna(0)
            else:
                primary_df["pnl"] = 0

        # compute synthetic pnl for shadows
        ml_df = compute_shadow_pnl(ml_df)
        alt_df = compute_shadow_pnl(alt_df)

        stats_primary = compute_stats(primary_df, pnl_col="pnl")
        stats_ml = compute_stats(ml_df, pnl_col="pnl")
        stats_alt = compute_stats(alt_df, pnl_col="pnl")

        score_primary = compute_evolution_score(stats_primary, self.cfg)
        score_ml = compute_evolution_score(stats_ml, self.cfg)
        score_alt = compute_evolution_score(stats_alt, self.cfg)

        leaderboard = [
            ("primary", score_primary, stats_primary),
            ("shadow_ml", score_ml, stats_ml),
            ("shadow_alt", score_alt, stats_alt),
        ]

        leaderboard_sorted = sorted(leaderboard, key=lambda x: x[1], reverse=True)

        # Save outputs
        out = {
            "scores": {
                "primary": score_primary,
                "shadow_ml": score_ml,
                "shadow_alt": score_alt,
            },
            "stats": {
                "primary": stats_primary,
                "shadow_ml": stats_ml,
                "shadow_alt": stats_alt,
            },
            "leaderboard": [x[0] for x in leaderboard_sorted],
        }

        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

        # CSV leaderboard
        rows = []
        for name, score, stats in leaderboard_sorted:
            row = {"bot": name, "score": score}
            row.update(stats)
            rows.append(row)

        pd.DataFrame(rows).to_csv(self.out_csv, index=False)

        log.info("Phase 99 Scoreboard written to %s", self.out_json)
        log.info("Phase 99 Leaderboard CSV written to %s", self.out_csv)

        # Telegram alert if shadow beats primary
        if (
            leaderboard_sorted[0][0] != "primary"
            and leaderboard_sorted[0][1] >= self.cfg.promotion_threshold
        ):
            if self.cfg.alert_enabled:
                try:
                    notify(
                        f"🔥 Shadow bot outperforming primary!\n"
                        f"Leader: {leaderboard_sorted[0][0]}\n"
                        f"Score: {leaderboard_sorted[0][1]:.4f}",
                        kind="system",
                        meta=out,
                    )
                except Exception:
                    log.error("Failed sending Telegram alert.")

        return out
