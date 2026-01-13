from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import yaml

@dataclass
class AppConfig:
    data: dict

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data=data)

    def y(self, key: str, default: Any = None) -> Any:
        cur = self.data
        for part in key.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
