from __future__ import annotations
import os
import glob
import time
import json
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai.execution.calibration import ExecCalibrator
from ai.envs.execution_env import ExecutionEnv

# --- Optional Telegram alerts ---
load_dotenv()
try:
    from utils.telegram_notifier import TelegramNotifier
except Exception:
    TelegramNotifier = None


@dataclass
class OnlineConfig:
    seed: int
    live_logs_glob: str
    cache_dir: str
    models_dir: str
    tensorboard_log: str
    base_model_path: str
    timesteps: int
    keep_last_n: int
    min_new_rows: int
    sleep_minutes: int
    env_cfg: Dict[str, Any]
    sim_cfg: Dict[str, Any]
    rew_cfg: Dict[str, Any]
    alerts_enabled: bool


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    sym_cnt = df["symbol"].nunique() if "symbol" in df else 0
    fill_rate = float(df.get("filled", pd.Series(dtype=float)).mean()) if "filled" in df else float("nan")
    return {"rows": int(len(df)), "symbols": int(sym_cnt), "fill_rate": fill_rate}


def _load_live_logs(glob_path: str) -> pd.DataFrame:
    """Loads live execution CSV logs written by LiveExecutionManager."""
    files = sorted(glob.glob(glob_path))
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Skipping unreadable log: {f} ({e})")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalize timestamp and numeric types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.Timestamp.utcnow()

    for c in ["qty", "mid", "spread", "fill_price", "latency_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive status column
    if "filled" in df.columns:
        df["status"] = np.where(df["filled"].astype(bool) & df["fill_price"].notna(), "filled", "open")
    else:
        df["status"] = "open"

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower().map(lambda x: "buy" if x.startswith("b") else "sell")

    # Ensure required columns
    for col in ["symbol", "side", "qty", "mid", "spread", "fill_price", "status"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.dropna(subset=["symbol", "side", "qty", "mid", "spread"], how="any")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _write_cache(df: pd.DataFrame, cache_dir: str) -> str:
    _ensure_dirs(cache_dir)
    out_path = os.path.join(cache_dir, f"live_merged_{_timestamp()}.parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def _merge_env_cfg(base: Dict[str, Any], sim: Dict[str, Any], rew: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**base}
    impact = sim.get("impact", {})
    merged.update({
        "passive_fill_prob_at_spread": sim.get("passive_fill_prob_at_spread", 0.35),
        "improve_fill_boost": sim.get("improve_fill_boost", 0.4),
        "join_fill_boost": sim.get("join_fill_boost", 0.15),
        "a_bps": impact.get("a_bps", 3.5),
        "sigma_bps": impact.get("sigma_bps", 2.0),
        "market_slippage_extra_bps": sim.get("market_slippage_extra_bps", 1.0),
        "fill_lat_ms_mean": sim.get("fill_lat_ms_mean", 120),
        "fill_lat_ms_std": sim.get("fill_lat_ms_std", 80),
        "cancel_prob": sim.get("cancel_prob", 0.08),
    })
    merged.update({
        "shortfall_weight": rew.get("shortfall_weight", 1.0),
        "inventory_weight": rew.get("inventory_weight", 5e-4),
        "latency_weight": rew.get("latency_weight", 1e-3),
        "fill_bonus": rew.get("fill_bonus", 0.05),
    })
    return merged


class OnlineRetrainer:
    def __init__(self, cfg: OnlineConfig):
        self.cfg = cfg
        self.alerts = TelegramNotifier() if cfg.alerts_enabled and TelegramNotifier else None
        _ensure_dirs(cfg.cache_dir, cfg.models_dir, os.path.dirname(cfg.tensorboard_log))

    # ------------------------------------------------------------------

    def _prune_checkpoints(self):
        """Keep only the latest N model checkpoints."""
        model_files = sorted(glob.glob(os.path.join(self.cfg.models_dir, "phase17_online_*.zip")))
        if len(model_files) <= self.cfg.keep_last_n:
            return
        for f in model_files[:-self.cfg.keep_last_n]:
            try:
                os.remove(f)
            except Exception:
                pass

    # ------------------------------------------------------------------

    def run_once(self) -> Optional[str]:
        """Run one retraining cycle using fresh live logs."""
        start_time = time.time()
        df = _load_live_logs(self.cfg.live_logs_glob)
        stats = _summarize_df(df)

        if stats["rows"] == 0 or stats["rows"] < self.cfg.min_new_rows:
            msg = f"ℹ️ Online retrain skipped: rows={stats['rows']} (<{self.cfg.min_new_rows})"
            print(msg)
            if self.alerts:
                self.alerts.send_message(msg)
            return None

        # Notify start
        if self.alerts:
            self.alerts.send_cycle_start(stats)

        cache_path = _write_cache(df, self.cfg.cache_dir)
        calibrator = ExecCalibrator(spread_floor_bps=self.cfg.env_cfg.get("spread_floor_bps", 0.2))
        params = calibrator.fit(df)

        calib_json = os.path.join(self.cfg.cache_dir, f"calibration_{_timestamp()}.json")
        with open(calib_json, "w") as f:
            json.dump(params, f, indent=2)

        env_cfg_merged = _merge_env_cfg(self.cfg.env_cfg, self.cfg.sim_cfg, self.cfg.rew_cfg)
        env_cfg_merged.update({
            "a_bps": params["impact"]["a_bps"],
            "sigma_bps": params["impact"]["sigma_bps"],
            "fill_lat_ms_mean": params["latency"]["mean_ms"],
            "fill_lat_ms_std": params["latency"]["std_ms"],
            "passive_fill_prob_at_spread": params["passive_fill_prob"],
            "spread_floor_bps": params["spread_floor_bps"],
        })

        def make_env():
            return ExecutionEnv(df, config=env_cfg_merged)

        vec_env = DummyVecEnv([make_env for _ in range(4)])

        # Load or initialize PPO
        if os.path.exists(self.cfg.base_model_path):
            model = PPO.load(self.cfg.base_model_path, env=vec_env)
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=min(1024, self.cfg.timesteps),
                batch_size=256,
                learning_rate=3e-4,
                ent_coef=0.01,
                verbose=1,
                seed=self.cfg.seed,
                tensorboard_log=self.cfg.tensorboard_log,
            )

        model.learn(total_timesteps=self.cfg.timesteps, progress_bar=False)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(self.cfg.models_dir, f"phase17_online_{ts}.zip")
        model.save(ckpt_path)
        shutil.copyfile(ckpt_path, self.cfg.base_model_path)
        self._prune_checkpoints()

        elapsed = time.time() - start_time
        print(f"✅ Online retraining complete: {ckpt_path} ({elapsed/60:.1f} min)")

        if self.alerts:
            self.alerts.send_cycle_end(ckpt_path, elapsed)

        return ckpt_path

    # ------------------------------------------------------------------

    def run_loop(self):
        """Run indefinitely, sleeping between retraining cycles."""
        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"❌ Online retraining error: {e}")
                if self.alerts:
                    self.alerts.send_cycle_error(e)
            time.sleep(self.cfg.sleep_minutes * 60)
