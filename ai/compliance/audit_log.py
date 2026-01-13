from __future__ import annotations
import json, os, time, hashlib
from typing import Any, Dict, Optional

class AuditLogger:
    def __init__(self, path: str, hash_chain: bool = True):
        self.path = path
        self.hash_chain = hash_chain
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._last_hash = None
        self._load_last_hash()

    def _load_last_hash(self):
        if not os.path.exists(self.path):
            self._last_hash = None
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                if end == 0:
                    self._last_hash = None
                    return
                # naive scan for last line
                f.seek(max(0, end - 1024))
                lines = f.read().splitlines()
                last = lines[-1]
                obj = json.loads(last.decode("utf-8"))
                self._last_hash = obj.get("_entry_hash")
        except Exception:
            self._last_hash = None

    def log(self, event: Dict[str, Any]):
        rec = {
            "ts": time.time(),
            **event
        }
        if self.hash_chain:
            payload = json.dumps(rec, sort_keys=True).encode("utf-8")
            base = hashlib.sha256(payload).hexdigest()
            chained = hashlib.sha256((base + (self._last_hash or "")).encode("utf-8")).hexdigest()
            rec["_entry_hash"] = chained
            self._last_hash = chained

        with open(self.path, "ab") as f:
            f.write((json.dumps(rec) + "\n").encode("utf-8"))
