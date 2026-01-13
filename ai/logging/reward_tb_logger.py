# ai/logging/reward_tb_logger.py
from __future__ import annotations
import os, csv, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

@dataclass
class LoggerConfig:
    enabled: bool = False
    log_dir: str = "runs/phase33"
    flush_every: int = 50
    fallback_csv: bool = True

class RewardLogger:
    """
    Tiny shim around TensorBoard's SummaryWriter with CSV fallback.
    - If torch/tensorboard is missing, it writes scalars to CSV in log_dir/scalars.csv
    - Public API:
        log(step: int, scalars: Dict[str, float])
        close()
    """
    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self._writer = None
        self._csv = None
        self._csv_path: Optional[Path] = None
        self._csv_fields: Optional[list[str]] = None
        self._log_dir: Optional[Path] = None
        if not cfg.enabled:
            return
        self._init_backend()

    def _init_backend(self):
        self._log_dir = Path(self.cfg.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # Try TB first
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._writer = SummaryWriter(log_dir=str(self._log_dir))
        except Exception:
            if self.cfg.fallback_csv:
                self._csv_path = self._log_dir / "scalars.csv"
                new_file = not self._csv_path.exists()
                self._csv = open(self._csv_path, "a", newline="", encoding="utf-8")
                self._csv_fields = ["step", "key", "value", "ts"]
                self._csv_writer = csv.DictWriter(self._csv, fieldnames=self._csv_fields)
                if new_file:
                    self._csv_writer.writeheader()
            else:
                # no-op logger
                pass

    def log(self, step: int, scalars: Dict[str, float]):
        if not self.cfg.enabled or not scalars:
            return
        if self._writer is not None:
            for k, v in scalars.items():
                try:
                    self._writer.add_scalar(k, float(v), global_step=step)
                except Exception:
                    # ignore bad values
                    pass
            if self.cfg.flush_every and step % self.cfg.flush_every == 0:
                try:
                    self._writer.flush()
                except Exception:
                    pass
        elif self._csv is not None:
            ts = time.time()
            for k, v in scalars.items():
                try:
                    self._csv_writer.writerow({"step": step, "key": k, "value": float(v), "ts": ts})
                except Exception:
                    pass
            if self.cfg.flush_every and step % self.cfg.flush_every == 0:
                try:
                    self._csv.flush()
                except Exception:
                    pass
        else:
            # disabled or no backend
            return

    def close(self):
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
        if self._csv is not None:
            try:
                self._csv.flush()
                self._csv.close()
            except Exception:
                pass
