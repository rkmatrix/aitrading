from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json, logging, time, datetime

from ai.policy.base import PolicyBundle
from ai.policy.validators import validate_bundle, PolicyValidationError
from ai.policy.providers import PROVIDER_MAP
from services.policy_registry import PolicyRegistry

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Optional Telegram notify shim
# ----------------------------------------------------------------------
try:
    from tools.telegram_alerts import notify
except Exception:
    def notify(msg: str, *, kind: str = "system", meta: dict | None = None):
        logger.info("[TELEGRAM:%s] %s", kind, msg)


# ----------------------------------------------------------------------
# Safe JSON serialization helper (date / Path tolerant)
# ----------------------------------------------------------------------
def _safe_json_dumps(obj: Any, **kwargs) -> str:
    def default_serializer(o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    return json.dumps(obj, default=default_serializer, **kwargs)


# ----------------------------------------------------------------------
# PolicySynchronizer
# ----------------------------------------------------------------------
class PolicySynchronizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.interval = int(cfg.get("interval_sec", 10))
        self.backoff = int(cfg.get("backoff_sec", 30))
        self.write_atomic = bool(cfg.get("write_atomic", True))
        self.out_dir = Path(cfg.get("output_dir", "data/runtime/policies"))
        self.registry = PolicyRegistry(cfg.get("registry_path", "data/runtime/policy_registry.json"))
        self.schemas = cfg.get("schemas", {})
        self.merge_strategy = cfg.get("merge_strategy", "prefer_latest_version")
        self.notifications = cfg.get("notifications", {"enabled": False, "channel": "system"})
        self.providers = self._build_providers(cfg.get("sources", []))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Provider setup
    # ------------------------------------------------------------------
    def _build_providers(self, src_cfgs: List[Dict[str, Any]]):
        providers = []
        for sc in src_cfgs:
            typ = sc.get("type")
            cls = PROVIDER_MAP.get(typ)
            if not cls:
                logger.warning("Unknown provider type '%s' – skipping", typ)
                continue
            try:
                providers.append(cls(sc))
            except Exception as ex:
                logger.error("Failed to init provider '%s': %s", typ, ex)
        return providers

    # ------------------------------------------------------------------
    # Normalize & validation
    # ------------------------------------------------------------------
    def _normalize(self, raw: Dict[str, Any]) -> PolicyBundle | None:
        try:
            name, version = validate_bundle(raw, self.schemas)
            schema = raw["schema"]
            kind = raw["kind"]
            payload = dict(raw)
            return PolicyBundle(
                name=name,
                version=str(version),
                schema=str(schema),
                kind=str(kind),
                payload=payload,
                source_id=raw.get("source_id", "unknown"),
            )
        except PolicyValidationError as ex:
            logger.error("Validation failed: %s", ex)
        except Exception as ex:
            logger.error("Normalize failed: %s", ex)
        return None

    # ------------------------------------------------------------------
    # Merge strategy
    # ------------------------------------------------------------------
    def _merge(self, bundles: List[PolicyBundle]) -> List[PolicyBundle]:
        by_name: Dict[str, List[PolicyBundle]] = {}
        for b in bundles:
            by_name.setdefault(b.name, []).append(b)
        result: List[PolicyBundle] = []
        for name, group in by_name.items():
            if len(group) == 1:
                result.append(group[0])
                continue
            if self.merge_strategy == "prefer_first_source":
                result.append(group[0])
            else:
                group_sorted = sorted(group, key=lambda x: (x.version, x.fetched_at_ts), reverse=True)
                result.append(group_sorted[0])
        return result

    # ------------------------------------------------------------------
    # Write to disk (atomic + safe JSON)
    # ------------------------------------------------------------------
    def _write_bundle(self, b: PolicyBundle) -> Tuple[Path, bool]:
        path = self.out_dir / b.output_filename()
        data = _safe_json_dumps(b.to_dict(), indent=2, sort_keys=True)

        if self.write_atomic:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)
        else:
            path.write_text(data, encoding="utf-8")

        reg_prev = self.registry.get(b.name)
        changed = (
            not reg_prev
            or reg_prev.get("checksum") != b.checksum()
            or reg_prev.get("version") != b.version
        )

        if changed:
            rec = self.registry.make_record(
                name=b.name,
                version=b.version,
                kind=b.kind,
                schema=b.schema,
                path=str(path),
                checksum=b.checksum(),
                source_id=b.source_id,
            )
            self.registry.set(rec)

        return path, changed

    # ------------------------------------------------------------------
    # Notify helper
    # ------------------------------------------------------------------
    def _notify(self, text: str) -> None:
        if self.notifications.get("enabled"):
            kind = self.notifications.get("channel", "system")
            try:
                notify(text, kind=kind)
            except Exception:
                logger.info("Notify: %s", text)

    # ------------------------------------------------------------------
    # Single pass
    # ------------------------------------------------------------------
    def once(self) -> int:
        collected: List[PolicyBundle] = []
        for p in self.providers:
            for raw in p.fetch():
                b = self._normalize(raw)
                if b:
                    collected.append(b)
        merged = self._merge(collected)
        updates = 0
        for b in merged:
            out_path, changed = self._write_bundle(b)
            if changed:
                updates += 1
                self._notify(f"📜 Policy updated: {b.name} v{b.version} → {out_path}")
        return updates

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        self._notify("🚀 Phase 41 Policy Synchronizer started")
        while True:
            try:
                n = self.once()
                if n == 0:
                    logger.debug("No policy changes detected")
                time.sleep(self.interval)
            except KeyboardInterrupt:
                self._notify("🛑 Policy Synchronizer stopped by user")
                break
            except Exception as ex:
                logger.exception("Sync loop error: %s", ex)
                time.sleep(self.backoff)
