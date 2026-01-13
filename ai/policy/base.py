from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import hashlib, json, time, datetime, logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyBundle:
    """
    Core data structure representing one policy document.

    Each policy is validated, normalized, and stored in runtime/policies.
    Includes automatic checksum generation that now handles datetime/date.
    """
    name: str
    version: str
    schema: str
    kind: str
    payload: Dict[str, Any]
    source_id: str
    fetched_at_ts: float = field(default_factory=lambda: time.time())

    # ------------------------------------------------------------------
    # Utility: Safe JSON serialization for payloads with date/datetime
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_dumps(obj: Any, **kwargs) -> str:
        """
        Dump JSON safely, converting date/time objects automatically.
        """
        def default_serializer(o):
            if isinstance(o, (datetime.date, datetime.datetime)):
                return o.isoformat()
            if isinstance(o, Path):
                return str(o)
            return str(o)  # fallback: stringify any unknown object

        return json.dumps(obj, default=default_serializer, **kwargs)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def checksum(self) -> str:
        """
        Compute SHA256 checksum of the payload using date-safe serialization.
        """
        data = self._safe_json_dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bundle to dictionary ready for registry writing.
        """
        base = {
            "name": self.name,
            "version": self.version,
            "schema": self.schema,
            "kind": self.kind,
            "payload": self.payload,
            "source_id": self.source_id,
            "fetched_at_ts": self.fetched_at_ts,
            "checksum": self.checksum(),
        }
        return json.loads(self._safe_json_dumps(base, sort_keys=True))

    def output_filename(self) -> str:
        """
        Create sanitized filename (e.g., risk_base__1.1.0.json)
        """
        safe_name = self.name.replace("/", "_").replace("\\", "_")
        safe_ver = self.version.replace("/", "_").replace("\\", "_")
        return f"{safe_name}__{safe_ver}.json"
