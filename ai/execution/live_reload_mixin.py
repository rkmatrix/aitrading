from __future__ import annotations
import json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LiveReloadMixin:
    _registry: dict = {}

    @classmethod
    def reload_policies(cls, registry: dict) -> None:
        cls._registry = dict(registry or {})
        logger.info("ExecutionAgent: reloaded registry with %d entries", len(cls._registry))

    @staticmethod
    def _load_json(path: str | Path) -> dict | None:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as ex:
            logger.error("Execution LiveReload read failed for %s: %s", path, ex)
            return None

    def get_exec_policy(self, name: str) -> dict | None:
        rec = self._registry.get(name)
        if not rec:
            return None
        return self._load_json(rec.get("path"))
