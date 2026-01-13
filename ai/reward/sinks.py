from __future__ import annotations
import csv, json, math, logging
from pathlib import Path
from typing import Dict, Optional, List

log = logging.getLogger("RewardSinks")

def _maybe_notify(kind: str, msg: str, meta: Optional[Dict] = None):
    try:
        from tools.telegram_alerts import notify  # your unified notifier
    except Exception:
        notify = None
    if notify:
        try:
            notify(msg, kind=kind, meta=meta or {})
        except Exception as e:
            log.warning(f"Notify failed: {e}")

class CsvSink:
    COLS = [
        "ts","symbol","price","position","realized_pnl","unrealized_pnl",
        "slippage","risk","equity","reward",
        "comp_pnl_realized","comp_pnl_unrealized","comp_drawdown","comp_slippage","comp_risk_dev",
        "trade_shares"
    ]
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_if_needed()

    def _init_if_needed(self):
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLS)

    def write(self, row: Dict):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, "") for k in self.COLS])

class SummarySink:
    def __init__(self, path: Path, z_alert_threshold: float = 3.0, notify_kind: str = "guardian", only_on_anomaly: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values: List[float] = []
        self.z_thr = float(z_alert_threshold)
        self.notify_kind = notify_kind
        self.only_on_anomaly = only_on_anomaly

    def update(self, reward: float):
        self.values.append(reward)

    def _zscore(self, x: float) -> float:
        if len(self.values) < 5: return 0.0
        mu = sum(self.values)/len(self.values)
        var = sum((v-mu)*(v-mu) for v in self.values)/len(self.values)
        std = math.sqrt(max(var, 1e-12))
        return (x - mu)/std if std > 0 else 0.0

    def flush(self, last_row: Dict):
        summary = {
            "count": len(self.values),
            "mean": (sum(self.values)/len(self.values)) if self.values else 0.0,
            "last_reward": self.values[-1] if self.values else 0.0
        }
        self.path.write_text(json.dumps(summary, indent=2))
        z = self._zscore(summary["last_reward"])
        if (not self.only_on_anomaly) or (abs(z) >= self.z_thr):
            _maybe_notify(self.notify_kind,
                f"Phase54 reward update: last={summary['last_reward']:.4f}, mean={summary['mean']:.4f}, z={z:.2f}",
                meta={"row": last_row})
