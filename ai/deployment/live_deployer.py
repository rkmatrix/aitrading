# ai/deployment/live_deployer.py
from __future__ import annotations
import csv, json, shutil, time, logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

class LivePolicyDeployer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger("LiveDeployer")
        self.source_csv = Path(cfg["source"]["training_log_csv"])
        self.src_dir = Path(cfg["source"]["policies_dir"])
        self.tgt_dir = Path(cfg["target"]["live_policy_dir"])
        self.deplog_csv = Path(cfg["outputs"]["deployment_log_csv"])
        self.registry_csv = Path(cfg["outputs"]["registry_csv"])

    def _load_training_log(self) -> pd.DataFrame:
        if not self.source_csv.exists():
            raise FileNotFoundError(f"Training log not found: {self.source_csv}")
        df = pd.read_csv(self.source_csv)
        if df.empty:
            raise ValueError("Training log empty.")
        return df

    def _select_best_policy(self, df: pd.DataFrame) -> str:
        metric = self.cfg["selection"].get("metric", "sharpe")
        min_score = float(self.cfg["selection"].get("min_score", 0.0))
        lookback_days = int(self.cfg["selection"].get("lookback_days", 7))
        prefer_newer = bool(self.cfg["selection"].get("prefer_newer", True))

        cutoff = time.time() - lookback_days * 86400
        df_recent = df[df["ts"] >= cutoff] if "ts" in df.columns else df
        df_recent = df_recent[df_recent[metric] >= min_score]
        if df_recent.empty:
            raise ValueError("No qualifying policies found in recent training log.")

        df_recent = df_recent.sort_values(
            by=[metric, "ts"] if prefer_newer else [metric],
            ascending=False
        )
        top_pid = df_recent.iloc[0]["policy_id"]
        top_row = df_recent.iloc[0].to_dict()
        self.logger.info(f"🏆 Selected top policy {top_pid} (metric={metric}, val={top_row[metric]:.3f})")
        return top_pid

    def _copy_bundle(self, pid: str) -> Dict[str, Any]:
        src = self.src_dir / pid
        if not src.exists():
            raise FileNotFoundError(f"Bundle folder not found: {src}")
        tgt = self.tgt_dir
        tgt.mkdir(parents=True, exist_ok=True)

        # purge old files
        for f in tgt.glob("*"):
            if f.is_file():
                f.unlink()

        # copy manifest + weights
        for fname in ["manifest.json", "model.pt"]:
            fp = src / fname
            if fp.exists():
                shutil.copy2(fp, tgt / fname)
        manifest = json.load(open(src / "manifest.json", "r", encoding="utf-8"))
        manifest["deployed_at"] = time.time()
        (tgt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _append_log(self, manifest: Dict[str, Any]):
        self.deplog_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.deplog_csv.exists()
        with open(self.deplog_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["ts","policy_id","version","metric","notes"])
            w.writerow([
                time.time(),
                manifest.get("policy_id"),
                manifest.get("version"),
                self.cfg["selection"].get("metric"),
                manifest.get("notes","")
            ])

    def _update_registry(self, manifest: Dict[str, Any]):
        reg = self.registry_csv
        reg.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if reg.exists():
            lines = reg.read_text(encoding="utf-8").splitlines()
        new_entry = f"{manifest['policy_id']},{self.tgt_dir},true,deployed {time.strftime('%Y-%m-%d %H:%M:%S')}"
        found = any(manifest["policy_id"] in line for line in lines)
        if not found:
            lines.append(new_entry)
            reg.write_text("\n".join(lines), encoding="utf-8")

    def deploy_latest(self) -> Dict[str, Any]:
        df = self._load_training_log()
        pid = self._select_best_policy(df)
        manifest = self._copy_bundle(pid)
        self._append_log(manifest)
        self._update_registry(manifest)
        self.logger.info(f"✅ Live policy updated → {self.tgt_dir}")
        return manifest
