"""
ai/orchestrator/policy_promoter.py
----------------------------------
Phase 52 – Policy Promotion Orchestrator

Scans models/policies/<PolicyName>/*/manifest.json to find candidates that
beat the current 'latest' bundle and promotes them by copying the candidate
folder into models/policies/<PolicyName>/latest (Windows-friendly).
"""

from __future__ import annotations
import json, logging, shutil, os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("PolicyPromoter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional Telegram notifier
try:
    from tools.telegram_alerts import notify
except Exception:  # pragma: no cover
    def notify(msg: str, *, kind: str = "system", meta: dict | None = None):
        logger.info(f"[TELEGRAM STUB] {msg}")


class PolicyPromoter:
    def __init__(self, cfg: dict):
        self.scan_root = Path(cfg.get("scan_root", "models/policies"))
        self.prom_log = Path(cfg.get("promotion_log", "data/reports/promotion_log.csv"))
        self.prom_log.parent.mkdir(parents=True, exist_ok=True)
        self.reward_min = float(cfg.get("reward_min", 0.55))
        self.reward_margin = float(cfg.get("reward_margin", 0.05))
        self.auto_restart = bool(cfg.get("auto_restart", False))
        self.tele_enabled = bool(cfg.get("telegram_alerts", False))
        self.templates = (cfg.get("telegram_templates") or {})

    # ---------------- filesystem helpers ----------------
    @staticmethod
    def _read_manifest(folder: Path) -> dict | None:
        m = folder / "manifest.json"
        if not m.exists():
            return None
        try:
            return json.loads(m.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Bad manifest at {m}: {e}")
            return None

    def _get_latest(self, policy_dir: Path) -> tuple[Path | None, dict | None]:
        latest = policy_dir / "latest"
        if latest.is_dir():
            return latest, self._read_manifest(latest)
        # support symlink if present
        if latest.exists() and latest.is_symlink():
            try:
                target = Path(os.readlink(latest))
                return target, self._read_manifest(target)
            except Exception:
                pass
        return None, None

    def _iter_candidates(self, policy_dir: Path) -> list[tuple[Path, dict]]:
        out: list[tuple[Path, dict]] = []
        for p in policy_dir.iterdir():
            if p.name.lower() == "latest":
                continue
            if p.is_dir():
                mf = self._read_manifest(p)
                if mf:
                    out.append((p, mf))
        return out

    @staticmethod
    def _safe_copytree(src: Path, dst: Path):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _promote(self, policy_dir: Path, from_dir: Path | None, to_dir: Path,
                 cur_reward: float | None, to_reward: float):
        latest = policy_dir / "latest"
        logger.info(f"Promoting {policy_dir.name}: {from_dir.name if from_dir else 'None'} → {to_dir.name}")
        self._safe_copytree(to_dir, latest)

        # --- Write promotion log ---
        row = ",".join([
            datetime.utcnow().isoformat(),
            policy_dir.name,
            (from_dir.name if from_dir else ""),
            to_dir.name,
            f"{cur_reward if cur_reward is not None else ''}",
            f"{to_reward}"
        ]) + "\n"
        if self.prom_log.exists():
            old = self.prom_log.read_text(encoding="utf-8")
            self.prom_log.write_text(old + row, encoding="utf-8")
        else:
            self.prom_log.write_text(row, encoding="utf-8")

        # --- Telegram alert ---
        if self.tele_enabled:
            delta = (to_reward - (cur_reward or 0.0))
            tmpl = self.templates.get(
                "promoted",
                "✅ {policy} promoted: {from_ver} → {to_ver} (Δ={delta:.3f})"
            )
            notify(
                tmpl.format(policy=policy_dir.name,
                            from_ver=(from_dir.name if from_dir else 'None'),
                            to_ver=to_dir.name,
                            delta=delta),
                kind="guardian"
            )

        # --- Create reload flag for executor ---
        flag = Path("data/runtime/reload_now.flag")
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(f"{policy_dir.name},{to_dir.name},{datetime.utcnow().isoformat()}", encoding="utf-8")
        logger.info(f"🪄 Reload flag written → {flag}")

        # --- Optional: auto_restart hook ---
        if self.auto_restart:
            logger.info("auto_restart=true (hook placeholder; add process control if desired)")


    # ---------------- decision logic ----------------
    def _choose_candidate(self, policy_dir: Path, cur_reward: float | None):
        best: tuple[Path, dict] | None = None
        for path, mf in self._iter_candidates(policy_dir):
            r = mf.get("reward_metric")
            acc = mf.get("accepted", 0)
            if r is None:
                continue
            r = float(r)
            if r < self.reward_min:
                continue
            if acc not in (1, True, "1"):
                continue
            if best is None or r > float(best[1].get("reward_metric", 0.0)):
                best = (path, mf)
        if best and self.tele_enabled:
            tmpl = self.templates.get("candidate_found", "🔎 {policy}: found candidate {to_ver} (reward {to_reward:.3f})")
            notify(tmpl.format(policy=policy_dir.name,
                               to_ver=best[0].name,
                               to_reward=float(best[1].get("reward_metric", 0.0))), kind="guardian")
        return best

    def _meets_margin(self, cand_reward: float, cur_reward: float | None) -> bool:
        if cur_reward is None:
            return cand_reward >= self.reward_min
        return cand_reward >= max(self.reward_min, cur_reward + self.reward_margin)

    # ---------------- public API ----------------
    def promote_policy(self, policy_dir: Path):
        latest_dir, latest_mf = self._get_latest(policy_dir)
        cur_reward = float(latest_mf.get("reward_metric", 0.0)) if latest_mf else None

        pick = self._choose_candidate(policy_dir, cur_reward)
        if not pick:
            if self.tele_enabled:
                tmpl = self.templates.get("skipped", "⏭️ {policy}: no candidate beats current (cur {cur_reward:.3f})")
                notify(tmpl.format(policy=policy_dir.name, cur_reward=(cur_reward or 0.0)), kind="guardian")
            logger.info(f"{policy_dir.name}: no qualifying candidate.")
            return

        cand_dir, cand_mf = pick
        cand_reward = float(cand_mf.get("reward_metric", 0.0))
        if not self._meets_margin(cand_reward, cur_reward):
            logger.info(f"{policy_dir.name}: candidate {cand_dir.name} ({cand_reward:.3f}) does not beat current margin.")
            return

        self._promote(policy_dir, latest_dir, cand_dir, cur_reward, cand_reward)

    def run_once(self):
        if not self.scan_root.exists():
            logger.warning(f"Scan root not found: {self.scan_root}")
            return
        for policy_dir in self.scan_root.iterdir():
            if policy_dir.is_dir():
                self.promote_policy(policy_dir)

    def run_forever(self, interval_sec: int = 1800):
        logger.info("🚀 Policy Promoter started …")
        while True:
            self.run_once()
            import time; time.sleep(interval_sec)
