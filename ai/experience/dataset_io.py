# ai/experience/dataset_io.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List
import pandas as pd
import numpy as np

def ensure_dirs(root: str):
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(root, "shards").mkdir(parents=True, exist_ok=True)

def pack_rows(transitions) -> pd.DataFrame:
    # transitions: Iterable[dict-like]
    rows = []
    for t in transitions:
        rows.append({
            "ts": t["ts"],
            "symbol": t["symbol"],
            "state": np.asarray(t["state"], dtype=np.float32),
            "action": np.asarray(t["action"], dtype=np.float32),
            "reward": float(t["reward"]),
            "next_state": np.asarray(t["next_state"], dtype=np.float32),
            "done": bool(t["done"]),
            "info": t.get("info", {}),
        })
    return pd.DataFrame(rows)

def write_parquet_sharded(df: pd.DataFrame, root: str, shard_idx: int):
    path = Path(root) / "shards" / f"exp_{shard_idx:06d}.parquet"
    df.to_parquet(path, index=False)

def stream_parquet(root: str, limit_files: int | None = None) -> Iterable[pd.DataFrame]:
    files = sorted(Path(root, "shards").glob("exp_*.parquet"))
    if limit_files is not None:
        files = files[:limit_files]
    for f in files:
        yield pd.read_parquet(f)
