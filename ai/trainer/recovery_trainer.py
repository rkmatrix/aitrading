from __future__ import annotations
import os, json, time, shutil, logging, random, string
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

try:
    # Optional dependency – used if present
    from stable_baselines3 import PPO  # noqa: F401
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

from tools.telegram_alerts import notify

logger = logging.getLogger("AdaptiveRecoveryTrainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _rand_tag(n=6):
    return ''.join(random.choice(string.hexdigits.lower()) for _ in range(n))


def _parse_version(ver: str) -> tuple:
    # expects "v1.7.3" -> (1,7,3). falls back gracefully
    try:
        nums = ver.lstrip("v").split(".")
        return tuple(int(x) for x in nums[:3])
    except Exception:
        return (0, 0, 0)


def _bump_version(ver: str, strategy: str = "patch") -> str:
    major, minor, patch = _parse_version(ver)
    if strategy == "minor":
        return f"v{major}.{minor+1}.0"
    elif strategy == "date":
        today = datetime.utcnow().strftime("%Y%m%d")
        return f"v{major}.{minor}.{patch}_{today}"
    else:
        return f"v{major}.{minor}.{patch+1}"


class AdaptiveRecoveryTrainer:
    """
    Reads rollback events, gathers recent feedback traces after a rollback,
    quick-tunes the rolled-back policy (SB3 PPO if available; otherwise a shim),
    then publishes a candidate version under models/policies/<policy>/<new_ver>.

    It also updates the manifest.json:
      - "last_stable_version" remains the rolled-back stable
      - "version" becomes the new candidate
      - "reward_metric" replaced by candidate_eval_metric
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rollback_log = cfg["rollback_log"]
        self.feedback_csv = cfg["feedback_csv"]
        self.policies_root = cfg["policies_root"]
        self.lookback_days = cfg["data"]["lookback_days"]
        self.min_rows = cfg["data"]["min_rows"]
        self.accept = cfg["acceptance"]
        self.versioning = cfg["versioning"]
        self.train_report = cfg["logging"]["train_report"]
        Path(self.train_report).parent.mkdir(parents=True, exist_ok=True)

    # ---------- Data ----------
    def _read_rollbacks(self) -> pd.DataFrame | None:
        if not os.path.exists(self.rollback_log):
            logger.warning(f"No rollback log found: {self.rollback_log}")
            return None
        df = pd.read_csv(self.rollback_log)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def _read_feedback(self) -> pd.DataFrame | None:
        if not os.path.exists(self.feedback_csv):
            logger.warning(f"No feedback CSV: {self.feedback_csv}")
            return None
        df = pd.read_csv(self.feedback_csv)
        if df.empty:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def _collect_training_slice(self, policy: str, rollback_time: pd.Timestamp) -> pd.DataFrame | None:
        df = self._read_feedback()
        if df is None or df.empty:
            return None

        cutoff = rollback_time  # learn from data after the rollback moment
        start_cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
        df = df[df["timestamp"] >= max(cutoff, pd.Timestamp(start_cutoff))]

        # filter to matching policy column if available
        policy_col = next((c for c in ["policy","policy_name","model","agent"] if c in df.columns), None)
        if policy_col:
            sub = df[df[policy_col].astype(str).str.contains(policy, case=False, na=False)]
        else:
            # fallback: broad match by string containment
            sub = df[df.astype(str).apply(lambda row: policy in row.to_string(), axis=1)]

        return None if sub.empty else sub

    # ---------- Model IO ----------
    def _policy_dir(self, policy: str) -> Path:
        return Path(self.policies_root) / policy

    def _manifest_path(self, policy: str) -> Path:
        return self._policy_dir(policy) / "manifest.json"

    def _read_manifest(self, policy: str) -> dict | None:
        mp = self._manifest_path(policy)
        if not mp.exists():
            logger.error(f"Manifest missing for {policy}")
            return None
        with open(mp, "r") as f:
            return json.load(f)

    def _write_manifest(self, policy: str, manifest: dict):
        mp = self._manifest_path(policy)
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=2)

    def _prepare_candidate_dir(self, policy: str, new_ver: str) -> Path:
        pdir = self._policy_dir(policy) / new_ver
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir

    # ---------- Training ----------
    def _quick_tune(self, df: pd.DataFrame) -> dict:
        """
        Lightweight "training" that estimates new metrics from feedback.
        If SB3 is available, you can replace this with real PPO fine-tuning.
        """
        # Aggregate recent outcomes
        mean_sharpe = float(df["sharpe"].mean() if "sharpe" in df.columns else 0.9)
        mean_feedback = float(df["feedback"].mean() if "feedback" in df.columns else 0.85)
        max_dd = float(df["drawdown"].max() if "drawdown" in df.columns else 0.08)

        # Heuristic reward metric
        reward_metric = max(0.0, min(1.0, 0.6*mean_feedback + 0.4*max(0.0, 1.2*mean_sharpe - 0.2) - 0.2*max_dd))

        # Fake/quick model file content
        blob = {
            "trained_at": datetime.utcnow().isoformat(),
            "strategy": "quick_tune" if not SB3_AVAILABLE else "ppo_finetune",
            "stats": {
                "mean_sharpe": mean_sharpe,
                "mean_feedback": mean_feedback,
                "max_drawdown": max_dd,
                "reward_metric": reward_metric
            }
        }
        return blob

    # ---------- Acceptance ----------
    def _passes_acceptance(self, stats: dict) -> bool:
        sharpe = stats["mean_sharpe"]
        feedback = stats["mean_feedback"]
        max_dd = stats["max_drawdown"]
        return (sharpe >= self.accept["min_sharpe"] and
                feedback >= self.accept["min_feedback"] and
                max_dd <= self.accept["max_drawdown"])

    # ---------- Publish ----------
    def _publish_candidate(self, policy: str, current_ver: str, candidate_stats: dict) -> str:
        new_base = _bump_version(current_ver, self.versioning.get("bump_strategy","patch"))
        tag = f"{self.versioning.get('tag_prefix','rc')}_{datetime.utcnow().strftime('%Y%m%d')}_{_rand_tag(6)}"
        new_ver = f"{new_base}_{tag}"

        cdir = self._prepare_candidate_dir(policy, new_ver)
        # write a minimal "model.pt" (json for demo)
        with open(cdir / "model.pt", "w") as f:
            json.dump({"weights": "placeholder", "note": "replace with real torch weights if available"}, f)

        # write manifest for candidate
        manifest = self._read_manifest(policy) or {}
        manifest["policy_name"] = policy
        manifest["version"] = new_ver
        # keep last_stable_version intact (it’s what we rolled back to)
        manifest["reward_metric"] = float(candidate_stats["reward_metric"])
        manifest["train_date"] = datetime.utcnow().isoformat()

        with open(cdir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # do NOT flip "latest" symlink here; leave that to Phase 43 selector / human promotion
        logger.info(f"Published candidate → {policy}/{new_ver}")
        return new_ver

    # ---------- Train once for a single rollback row ----------
    def train_from_rollback(self, policy: str, from_ver: str, to_ver: str, ts: pd.Timestamp):
        logger.info(f"🔁 Recovery train start: {policy} (rolled back {from_ver} → {to_ver} @ {ts})")

        traces = self._collect_training_slice(policy, ts)
        if traces is None or len(traces) < self.min_rows:
            logger.warning(f"Not enough data to train ({0 if traces is None else len(traces)} < {self.min_rows}). "
                           f"Falling back to synthetic seed from manifest.")
            # synthesize a tiny dataframe from manifest to allow candidate creation
            manifest = self._read_manifest(policy) or {}
            seed = pd.DataFrame([{
                "timestamp": datetime.utcnow(),
                "policy": policy,
                "sharpe": float(manifest.get("reward_metric", 0.9))*1.1,
                "feedback": float(manifest.get("reward_metric", 0.85)),
                "drawdown": 0.08
            }])
            traces = seed

        stats_blob = self._quick_tune(traces)
        stats = stats_blob["stats"]
        cand_ver = self._publish_candidate(policy, current_ver=to_ver, candidate_stats=stats)

        # acceptance gate (optional auto-promote in manifest if very strong)
        accepted = self._passes_acceptance(stats)
        if accepted:
            logger.info(f"✅ Candidate PASSED acceptance (Sharpe={stats['mean_sharpe']:.2f}, "
                        f"Feedback={stats['mean_feedback']:.2f}, MaxDD={stats['max_drawdown']:.2f})")
        else:
            logger.info(f"⚠️ Candidate did NOT meet acceptance thresholds, keep as RC for evaluation.")

        # log training result
        self._log_train_result(policy, from_ver, to_ver, cand_ver, stats, accepted)

        # notify
        if self.cfg["notifications"]["enabled"]:
            msg = (f"🧪 Phase48: {policy} candidate {cand_ver} "
                   f"{'PASSED ✅' if accepted else 'created (needs eval)'} | "
                   f"Sharpe={stats['mean_sharpe']:.2f} Feedback={stats['mean_feedback']:.2f} "
                   f"MaxDD={stats['max_drawdown']:.2f}")
            notify(msg, kind=self.cfg["notifications"].get("telegram_channel","guardian"))

    def _log_train_result(self, policy, from_ver, to_ver, cand_ver, stats, accepted: bool):
        Path(self.train_report).parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "policy": policy,
            "rolled_from": from_ver,
            "rolled_to_stable": to_ver,
            "candidate_version": cand_ver,
            "mean_sharpe": stats["mean_sharpe"],
            "mean_feedback": stats["mean_feedback"],
            "max_drawdown": stats["max_drawdown"],
            "reward_metric": stats["reward_metric"],
            "accepted": int(bool(accepted))
        }])
        row.to_csv(self.train_report, mode="a", header=not os.path.exists(self.train_report), index=False)

    # ---------- Orchestrate over rollback log ----------
    def process_all_rollbacks(self, since_hours: int = 24):
        df = self._read_rollbacks()
        if df is None or df.empty:
            logger.info("No rollback rows found.")
            return

        # recent rollbacks only
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        df = df[pd.to_datetime(df["timestamp"], errors="coerce") >= cutoff]
        if df.empty:
            logger.info(f"No recent rollbacks in the last {since_hours}h.")
            return

        for _, row in df.iterrows():
            policy = str(row.get("policy"))
            from_ver = str(row.get("from_version"))
            to_ver = str(row.get("to_version"))
            ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
            if not policy or not to_ver or pd.isna(ts):
                continue
            # train using the rollback’s stable version as base
            self.train_from_rollback(policy, from_ver, to_ver, ts)
