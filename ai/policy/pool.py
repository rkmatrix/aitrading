# ai/policy/pool.py
from __future__ import annotations
import json, uuid, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

@dataclass
class PolicyBundle:
    policy_id: str
    path: Path
    manifest: Dict[str, Any]
    has_weights: bool
    created_at: float = field(default_factory=lambda: time.time())
    lineage: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_dir(cls, d: Path) -> "PolicyBundle":
        d = Path(d)
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {d}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        pid = manifest.get("policy_id") or manifest.get("name") or d.name
        has_weights = (d / "model.pt").exists() or (d / "weights.bin").exists()
        lineage = manifest.get("lineage", {})
        return cls(policy_id=str(pid), path=d, manifest=manifest, has_weights=has_weights, lineage=lineage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "path": str(self.path),
            "has_weights": self.has_weights,
            "manifest": self.manifest,
            "created_at": self.created_at,
            "lineage": self.lineage,
        }

@dataclass
class PolicyPool:
    bundles: Dict[str, PolicyBundle] = field(default_factory=dict)
    max_size: int = 24

    def add(self, b: PolicyBundle) -> None:
        self.bundles[b.policy_id] = b
        self._trim()

    def remove(self, policy_id: str) -> None:
        self.bundles.pop(policy_id, None)

    def get(self, policy_id: str) -> Optional[PolicyBundle]:
        return self.bundles.get(policy_id)

    def list_ids(self) -> List[str]:
        return list(self.bundles.keys())

    def _trim(self) -> None:
        if len(self.bundles) <= self.max_size: 
            return
        # remove oldest by created_at
        items = sorted(self.bundles.values(), key=lambda x: x.created_at)
        for b in items[:-self.max_size]:
            self.remove(b.policy_id)

    @classmethod
    def from_directory(cls, root: Path, max_size: int = 24) -> "PolicyPool":
        pool = cls(max_size=max_size)
        if not Path(root).exists():
            return pool
        for sub in Path(root).iterdir():
            if not sub.is_dir(): 
                continue
            try:
                pool.add(PolicyBundle.load_from_dir(sub))
            except Exception:
                # skip invalid
                continue
        return pool

    def spawn_variant(self, parent: PolicyBundle, new_manifest: Dict[str, Any], suffix: str = "") -> PolicyBundle:
        # This just tracks the variant in memory; writing to disk is Engine's job
        new_pid = f"{parent.policy_id}_v{uuid.uuid4().hex[:6]}{suffix}"
        nm = json.loads(json.dumps(new_manifest))
        nm["policy_id"] = new_pid
        nm["lineage"] = {
            "parent": parent.policy_id,
            "ts": time.time(),
            **nm.get("lineage", {})
        }
        return PolicyBundle(policy_id=new_pid, path=Path(""), manifest=nm, has_weights=False, lineage=nm["lineage"])
