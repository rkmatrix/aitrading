# ai/trainer/utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

def find_untrained_variants(policy_dir: Path) -> List[Path]:
    """
    Returns policy variant folders missing model.pt (metadata-only).
    """
    missing = []
    for sub in Path(policy_dir).iterdir():
        if not sub.is_dir(): continue
        if not (sub / "manifest.json").exists(): continue
        if not (sub / "model.pt").exists():
            missing.append(sub)
    return missing

def append_training_log(csv_path: Path, rows: List[dict]) -> None:
    import csv, os
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts","policy_id","score","sharpe","winrate","steps"])
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({
                "ts": r["ts"],
                "policy_id": r["policy_id"],
                "score": r["score"],
                "sharpe": r["sharpe"],
                "winrate": r["winrate"],
                "steps": r["steps"]
            })
