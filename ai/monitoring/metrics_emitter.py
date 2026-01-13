from __future__ import annotations
import csv, json, os, time
from typing import Dict, Any

class MetricsEmitter:
    def __init__(self, csv_path: str, jsonl_path: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        self._csv_initialized = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def emit(self, metrics: Dict[str, Any]):
        ts = time.time()
        rec = {"ts": ts, **metrics}

        # JSONL
        with open(self.jsonl_path, "ab") as f:
            f.write((json.dumps(rec) + "\n").encode("utf-8"))

        # CSV (flat only)
        keys = sorted(rec.keys())
        write_header = not self._csv_initialized
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
                self._csv_initialized = True
            w.writerow(rec)
