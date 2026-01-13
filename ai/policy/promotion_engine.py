# ai/policy/promotion_engine.py
from __future__ import annotations
import json, logging, shutil, time
from pathlib import Path
from dataclasses import dataclass
from tools.telegram_alerts import notify

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardEntry:
    name: str
    sharpe: float
    winrate: float
    drawdown: float
    version: str
    bundle_path: Path


class PolicyPromotionEngine:
    """
    Phase 100 – Automatic Policy Promotion Engine

    Responsibilities:
    -----------------
    • Read leaderboard (Phase 99 output)
    • Determine if a shadow policy beats the primary policy
    • Create new versioned bundle
    • Update manifest.json
    • Promote model to "primary"
    • Archive previous primary
    • Notify via Telegram
    """

    def __init__(
        self,
        leaderboard_path="data/reports/phase99_leaderboard.json",
        models_root="models/policies",
        primary_name="EquityRLPolicy",
        min_imp_sharpe=0.05,
        min_imp_winrate=0.02,
        min_imp_dd_reduction=0.03,
    ):
        self.leaderboard_path = Path(leaderboard_path)
        self.models_root = Path(models_root)
        self.primary_name = primary_name

        self.min_imp_sharpe = min_imp_sharpe
        self.min_imp_winrate = min_imp_winrate
        self.min_imp_dd = min_imp_dd_reduction

        self.primary_path = self.models_root / primary_name

    # ---------------------------------------------------------------------
    # Load leaderboard (Phase 99)
    # ---------------------------------------------------------------------
    def load_leaderboard(self) -> list[LeaderboardEntry]:
        if not self.leaderboard_path.exists():
            raise FileNotFoundError(f"Leaderboard not found: {self.leaderboard_path}")

        with open(self.leaderboard_path, "r") as f:
            data = json.load(f)

        entries = []
        for row in data:
            entries.append(
                LeaderboardEntry(
                    name=row["name"],
                    sharpe=row["sharpe"],
                    winrate=row["winrate"],
                    drawdown=row["drawdown"],
                    version=row["version"],
                    bundle_path=Path(row["bundle_path"]),
                )
            )
        return entries

    # ---------------------------------------------------------------------
    def _read_manifest(self, path: Path) -> dict:
        manifest = path / "manifest.json"
        with open(manifest, "r") as f:
            return json.load(f)

    # ---------------------------------------------------------------------
    # Eligibility check
    # ---------------------------------------------------------------------
    def is_eligible(self, primary: LeaderboardEntry, challenger: LeaderboardEntry) -> bool:
        d_sharpe = challenger.sharpe - primary.sharpe
        d_win = challenger.winrate - primary.winrate
        d_dd = primary.drawdown - challenger.drawdown  # lower is better

        logger.info(
            f"Comparing challenger={challenger.name} vs primary={primary.name} | "
            f"dSharpe={d_sharpe:.3f}, dWin={d_win:.3f}, dDD={d_dd:.3f}"
        )

        if d_sharpe < self.min_imp_sharpe:
            return False
        if d_win < self.min_imp_winrate:
            return False
        if d_dd < self.min_imp_dd:
            return False

        return True

    # ---------------------------------------------------------------------
    # Promotion workflow
    # ---------------------------------------------------------------------
    def promote(self, challenger: LeaderboardEntry):
        notify(f"🚀 Policy Promotion starting: {challenger.name} ({challenger.version})", kind="system")

        # Archive old primary
        archive_root = self.primary_path / "archive"
        archive_root.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_dest = archive_root / f"{challenger.name}_old_{ts}"
        shutil.move(str(self.primary_path), str(archive_dest))

        # Copy challenger bundle → new primary
        shutil.copytree(challenger.bundle_path, self.primary_path)

        # Update manifest version
        manifest = self._read_manifest(self.primary_path)
        old_ver = manifest.get("version", "v0")
        major, minor, patch = map(int, old_ver.strip("v").split("."))

        new_version = f"v{major}.{minor+1}.0"
        manifest["version"] = new_version

        with open(self.primary_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        notify(f"✅ Policy Promotion completed → {new_version}", kind="system")
        logger.info(f"Promoted {challenger.name} to primary model version {new_version}")

    # ---------------------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------------------
    def run(self):
        lb = self.load_leaderboard()
        if len(lb) < 2:
            logger.warning("Not enough models to compare.")
            return False

        # Primary model entry
        primary_entry = None
        for e in lb:
            if e.name == self.primary_name:
                primary_entry = e
                break

        if primary_entry is None:
            raise RuntimeError("Primary model not found in leaderboard.")

        # Identify top challenger
        challengers = [e for e in lb if e.name != self.primary_name]
        challengers = sorted(challengers, key=lambda x: x.sharpe, reverse=True)
        challenger = challengers[0]

        logger.info(f"Best challenger = {challenger.name}")

        if self.is_eligible(primary_entry, challenger):
            self.promote(challenger)
            return True
        else:
            logger.info("No challenger meets the promotion criteria.")
            return False
