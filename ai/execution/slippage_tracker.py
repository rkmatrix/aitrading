# ai/execution/slippage_tracker.py

import pandas as pd
from datetime import datetime
from pathlib import Path

class SlippageTracker:
    def __init__(self, path="data/execution_logs/slippage_log.csv"):
        self.path = path

    def log(self, order_id, expected_price, fill_price, qty):
        slip = (fill_price - expected_price) / expected_price * 100
        record = {
            "timestamp": datetime.utcnow(),
            "order_id": order_id,
            "expected_price": expected_price,
            "fill_price": fill_price,
            "slippage_pct": slip,
            "qty": qty
        }
        df = pd.DataFrame([record])

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 🧠 Always write header if file does not exist OR is empty
        write_header = not path.exists() or path.stat().st_size == 0

        df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        # 🧩 Double-check that file is readable and valid (sanity check)
        try:
            check_df = pd.read_csv(path)
            if "slippage_pct" not in check_df.columns:
                # Recreate the file with proper headers if needed
                df.to_csv(path, index=False)
        except Exception:
            # Recover gracefully if file is corrupted
            df.to_csv(path, index=False)

        return float(slip)
