from __future__ import annotations
from typing import Dict, Any, Iterable
import json, logging, urllib.request

logger = logging.getLogger(__name__)

class HttpJsonProvider:
    def __init__(self, cfg: Dict[str, Any]):
        self.id = cfg.get("id", "http")
        self.url = cfg["url"]
        self.timeout = int(cfg.get("timeout_sec", 10))

    def fetch(self) -> Iterable[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                items = data.values() if any(isinstance(v, dict) for v in data.values()) else [data]
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for item in items:
                if isinstance(item, dict):
                    item.setdefault("source_id", self.id)
                    yield item
        except Exception as ex:
            logger.error("HttpJsonProvider failed for %s: %s", self.url, ex)
