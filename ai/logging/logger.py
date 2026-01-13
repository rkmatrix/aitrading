import csv
import os
from pathlib import Path
from typing import Dict, Optional


class CSVLogger:
    def __init__(self, out_dir: str = "artifacts/phase7_logs", filename: str = "online_metrics.csv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._header_written = self.path.exists()

    def log(self, row: Dict):
        if not self._header_written:
            with self.path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
            self._header_written = True
        else:
            with self.path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)


class WandbLogger:
    def __init__(self, enabled: bool, project: Optional[str] = None, run_name: Optional[str] = None):
        self.enabled = enabled
        self.wb = None
        if not enabled:
            return
        try:
            import wandb
            self.wb = wandb
            self.wb.init(project=project or "ai-tradebot-phase7", name=run_name, reinit=True)
        except Exception as e:
            print(f"⚠️ wandb disabled: {e}")
            self.enabled = False

    def log(self, row: Dict):
        if not self.enabled or self.wb is None:
            return
        try:
            self.wb.log(row)
        except Exception:
            pass


class MLflowLogger:
    def __init__(self, enabled: bool, run_name: Optional[str] = None):
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment("ai-tradebot-phase7")
            self.mlflow.start_run(run_name=run_name)
        except Exception as e:
            print(f"⚠️ mlflow disabled: {e}")
            self.enabled = False

    def log(self, row: Dict):
        if not self.enabled or self.mlflow is None:
            return
        try:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, float(v))
        except Exception:
            pass

    def end(self):
        if self.enabled and self.mlflow:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
