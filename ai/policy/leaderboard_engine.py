# ai/policy/leaderboard_engine.py
from __future__ import annotations
import json, logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyPerf:
    name: str
    sharpe: float
    winrate: float
    drawdown: float
    profit_factor: float
    consistency: float
    version: str
    bundle_path: Path


class PolicyLeaderboardEngine:
    """
    Phase 99 – Policy Leaderboard Generator
    ---------------------------------------
    Reads performance reports from:
        data/reports/policy_perf/<policy_name>.json

    Reads metadata from:
        models/policies/<policy_name>/manifest.json

    Outputs:
        data/reports/phase99_leaderboard.json

    This drives Phase 100 Promotion Engine.
    """

    def __init__(
        self,
        models_root="models/policies",
        perf_root="data/reports/policy_perf",
        out_path="data/reports/phase99_leaderboard.json",
        min_history=20
    ) -> None:
        self.models_root = Path(models_root)
        self.perf_root = Path(perf_root)
        self.out_path = Path(out_path)
        self.min_history = min_history

        self.out_path.parent.mkdir(exist_ok=True, parents=True)

    # ---------------------------------------------------------------------
    def _load_manifest(self, policy_path: Path) -> dict | None:
        manifest_file = policy_path / "manifest.json"
        if not manifest_file.exists():
            logger.warning(f"Manifest missing for {policy_path.name}")
            return None
        return json.loads(manifest_file.read_text())

    # ---------------------------------------------------------------------
    def _load_perf(self, policy_name: str) -> dict | None:
        perf_file = self.perf_root / f"{policy_name}.json"
        if not perf_file.exists():
            logger.warning(f"Performance file not found: {perf_file}")
            return None
        return json.loads(perf_file.read_text())

    # ---------------------------------------------------------------------
    def compute_metrics(self, perf: dict) -> tuple:
        """Computes Sharpe, Winrate, Drawdown, Profit Factor, Consistency."""
        rets = np.array(perf.get("returns", []), dtype=float)
        wins = np.array(perf.get("win_flags", []), dtype=float)

        if len(rets) < self.min_history:
            raise ValueError("Not enough history for performance rating.")

        # Sharpe ratio
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8)

        # Winrate
        winrate = np.mean(wins)

        # Max drawdown
        cum = np.cumsum(rets)
        drawdown = float(np.max(cum) - np.min(cum))

        # Profit factor
        gains = rets[rets > 0].sum()
        losses = -rets[rets < 0].sum()
        profit_factor = gains / (losses + 1e-8)

        # Consistency score (0–1)
        consistency = np.mean(np.abs(rets) < (np.std(rets) * 1.5))

        return sharpe, winrate, drawdown, profit_factor, consistency

    # ---------------------------------------------------------------------
    def run(self) -> list[PolicyPerf]:
        logger.info("🔎 Phase 99 — Building Policy Leaderboard")

        entries: list[PolicyPerf] = []

        for policy_dir in self.models_root.iterdir():
            if not policy_dir.is_dir():
                continue

            name = policy_dir.name

            manifest = self._load_manifest(policy_dir)
            if manifest is None:
                continue

            version = manifest.get("version", "v0")

            perf = self._load_perf(name)
            if perf is None:
                continue

            try:
                sharpe, winrate, dd, pf, cons = self.compute_metrics(perf)
            except Exception as e:
                logger.warning(f"Skipping {name} due to insufficient data: {e}")
                continue

            entry = PolicyPerf(
                name=name,
                sharpe=sharpe,
                winrate=winrate,
                drawdown=dd,
                profit_factor=pf,
                consistency=cons,
                version=version,
                bundle_path=policy_dir,
            )
            entries.append(entry)

        # Sort by Sharpe DESC
        entries = sorted(entries, key=lambda x: x.sharpe, reverse=True)

        # Write leaderboard
        out_data = []
        for e in entries:
            out_data.append(
                dict(
                    name=e.name,
                    sharpe=e.sharpe,
                    winrate=e.winrate,
                    drawdown=e.drawdown,
                    profit_factor=e.profit_factor,
                    consistency=e.consistency,
                    version=e.version,
                    bundle_path=str(e.bundle_path),
                )
            )

        with open(self.out_path, "w") as f:
            json.dump(out_data, f, indent=4)

        logger.info(f"🏁 Leaderboard saved → {self.out_path}")
        return entries
