import os, csv
from datetime import datetime
from typing import Dict
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Telemetry:
    def __init__(self, log_dir: str, use_csv=True, use_tb=True, use_stdout=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.use_stdout = use_stdout
        self.tb = SummaryWriter(log_dir) if use_tb and SummaryWriter else None
        self.csv_path = os.path.join(log_dir, "metrics.csv") if use_csv else None
        if self.csv_path and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["ts", "step", "metric", "value"])

    def log(self, step: int, metrics: Dict[str, float]):
        ts = datetime.utcnow().isoformat()
        if self.use_stdout:
            print("[telemetry]", step, {k: round(v,6) for k,v in metrics.items()})
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                for k, v in metrics.items():
                    w.writerow([ts, step, k, float(v)])
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, float(v), step)

    def close(self):
        if self.tb:
            self.tb.close()
