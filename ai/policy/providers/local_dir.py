from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List
import json, logging, yaml

logger = logging.getLogger(__name__)

class LocalDirProvider:
    def __init__(self, cfg: Dict[str, Any]):
        self.id = cfg.get("id", "local")
        self.root = Path(cfg["root"]).resolve()
        self.patterns: List[str] = list(cfg.get("patterns", ["*.json"]))
        if not self.root.exists():
            logger.warning("LocalDirProvider: root does not exist: %s", self.root)

    def fetch(self) -> Iterable[Dict[str, Any]]:
        for pat in self.patterns:
            for fp in self.root.rglob(pat):
                try:
                    raw = fp.read_text(encoding="utf-8")
                    if fp.suffix.lower() == ".json":
                        data = json.loads(raw)
                    else:
                        data = yaml.safe_load(raw)
                    if isinstance(data, dict):
                        data.setdefault("source_id", self.id)
                        yield data
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item.setdefault("source_id", self.id)
                                yield item
                except Exception as ex:
                    logger.error("LocalDirProvider failed for %s: %s", fp, ex)
