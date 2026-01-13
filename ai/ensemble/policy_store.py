# ai/ensemble/policy_store.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Protocol, Tuple


class Policy(Protocol):
    """Minimal interface your policies must implement for blending."""
    name: str
    action_space: Any

    def predict(self, obs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        ...

    def load(self, path: str) -> None:
        ...


@dataclass
class PolicyMeta:
    name: str
    tag: str
    path: str
    regime_hint: Optional[str] = None
    notes: Optional[str] = None


class PolicyStore:
    """Registry for multiple trained policies (Momentum, MeanRev, Macro, etc.)."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._policies: Dict[str, Policy] = {}
        self._meta: Dict[str, PolicyMeta] = {}

    def register(self, key: str, policy: Policy, meta: PolicyMeta) -> None:
        if key in self._policies:
            raise ValueError(f"Policy key already exists: {key}")
        self._policies[key] = policy
        self._meta[key] = meta

    def keys(self) -> List[str]:
        return list(self._policies.keys())

    def get(self, key: str) -> Policy:
        return self._policies[key]

    def meta(self, key: str) -> PolicyMeta:
        return self._meta[key]

    def load_from_manifest(self, manifest_path: str, policy_factory) -> None:
        """
        Load policies declared in a JSON manifest.
        Example:
        {
            "policies": [
                {"key": "mom", "name": "MomentumPPO", "tag": "v11", "path": "models/mom_v11.zip", "regime_hint": "trend"},
                {"key": "mr",  "name": "MeanRevPPO",  "tag": "v8",  "path": "models/mr_v8.zip",  "regime_hint": "range"}
            ]
        }
        policy_factory(name) must return a Policy instance.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("policies", []):
            inst = policy_factory(p["name"])
            inst.load(p["path"])
            meta = PolicyMeta(
                name=p["name"],
                tag=p["tag"],
                path=p["path"],
                regime_hint=p.get("regime_hint")
            )
            self.register(p["key"], inst, meta)

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": k,
                "name": self._meta[k].name,
                "tag": self._meta[k].tag,
                "path": self._meta[k].path,
                "regime_hint": self._meta[k].regime_hint,
            }
            for k in self._policies
        ]
