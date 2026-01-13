# ai/evolution/archive.py
from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Dict, Any, Optional

class EvolutionArchive:
    def __init__(self, log_csv: Path):
        self.log_csv = Path(log_csv)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","event","policy_id","parent","score","notes"])

    def log_event(self, event: str, policy_id: str, score: Optional[float] = None, parent: Optional[str] = None, notes: str = ""):
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([time.time(), event, policy_id, parent or "", "" if score is None else score, notes])
