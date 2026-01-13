# SPDX-License-Identifier: MIT
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

# Ensure project root on sys.path when running directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ai.allocator.cap_rules import ExposureCaps, HardRisk, clamp_weights
from ai.allocator.policies.base import AllocationPolicy
from ai.allocator.policies.heuristic import HeuristicPolicy
from ai.allocator.policies.ppo_stub import PPOPolicyStub


@dataclass
class CoordinatorConfig:
    symbols: List[str]
    artifacts_alloc_dir: Path
    artifacts_latest_symlink: Path
    inputs_signals: Path
    inputs_balances: Path
    inputs_positions: Path
    handoff_targets: Path
    exposure_caps: ExposureCaps
    hard_risk: HardRisk
    policy_name: str
    policy_params: Dict[str, Any]
    logging_level: str = "INFO"


class CapitalAllocatorCoordinator:
    def __init__(self, cfg: CoordinatorConfig):
        self.cfg = cfg
        self.policy = self._make_policy(cfg.policy_name, cfg.policy_params)

        self.cfg.artifacts_alloc_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.handoff_targets.parent.mkdir(parents=True, exist_ok=True)

    def _make_policy(self, name: str, params: Dict[str, Any]) -> AllocationPolicy:
        name = (name or "heuristic").lower()
        if name == "heuristic":
            return HeuristicPolicy(**params)
        if name == "ppo":
            return PPOPolicyStub(**params)
        raise ValueError(f"Unknown policy: {name}")

    # ---- IO helpers ----
    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _read_inputs(self) -> Dict[str, Any]:
        signals = self._load_json(self.cfg.inputs_signals, default={})
        balances = self._load_json(self.cfg.inputs_balances, default={})
        positions = self._load_json(self.cfg.inputs_positions, default={})
        return {"signals": signals, "balances": balances, "positions": positions}

    # ---- Main step ----
    def compute_allocation(self) -> Dict[str, float]:
        data = self._read_inputs()

        # The policy returns raw target weights (may violate soft caps)
        raw_weights = self.policy.propose_weights(
            symbols=self.cfg.symbols,
            signals=data.get("signals", {}),
            balances=data.get("balances", {}),
            positions=data.get("positions", {}),
        )

        # Enforce exposure caps
        # NOTE: sector mapping can be injected later; use None for now
        clamped = clamp_weights(raw_weights, self.cfg.exposure_caps, symbol_to_sector=None)

        # Normalize to keep some cash floor if policy needs it handled here
        # (heuristic policy already tries to honor cash; this is a safeguard)
        gross = sum(abs(v) for v in clamped.values())
        if gross > 1.0:
            scale = 1.0 / gross
            clamped = {k: v * scale for k, v in clamped.items()}

        return clamped

    def _write_artifacts(self, weights: Dict[str, float]) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        snapshot = {
            "timestamp": ts,
            "weights": weights,
            "symbols": self.cfg.symbols,
            "source": "phase30",
        }
        out_path = self.cfg.artifacts_alloc_dir / f"alloc_{ts}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        # Update "latest" symlink/file
        try:
            tmp = self.cfg.artifacts_latest_symlink
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            tmp.symlink_to(out_path)
        except Exception:
            # On Windows, create a copy instead of a symlink
            with self.cfg.artifacts_latest_symlink.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)

        return out_path

    def _write_targets(self, weights: Dict[str, float]) -> Path:
        payload = {"targets": weights, "ts": time.time(), "source": "phase30"}
        with self.cfg.handoff_targets.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.cfg.handoff_targets

    def run_once(self) -> Dict[str, Any]:
        weights = self.compute_allocation()
        snap = self._write_artifacts(weights)
        tgt = self._write_targets(weights)
        return {"weights": weights, "artifact": str(snap), "targets": str(tgt)}
