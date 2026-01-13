from __future__ import annotations
import yaml, os, time
from typing import Any

class FlagStore:
    def __init__(self, path: str, auto_reload_seconds: int = 15):
        self.path = path
        self.auto_reload = auto_reload_seconds
        self._last_load = 0.0
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self._data = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}
        self._last_load = time.time()

    def get(self, dotted_path: str, default: Any = None) -> Any:
        now = time.time()
        if now - self._last_load >= self.auto_reload:
            self._load()
        cur = self._data
        for part in dotted_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur
