# ai/policy/perf_recorder.py
from __future__ import annotations
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PerformanceRecorder:
    """
    Central performance recorder for all RL policies.
    Writes:
        data/reports/policy_perf/<policy_name>.json

    Data format:
        {
            "returns": [...],
            "win_flags": [...]
        }
    """

    def __init__(self, perf_root="data/reports/policy_perf"):
        self.perf_root = Path(perf_root)
        self.perf_root.mkdir(parents=True, exist_ok=True)

    def _load(self, name: str) -> dict:
        path = self.perf_root / f"{name}.json"
        if not path.exists():
            return {"returns": [], "win_flags": []}
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.error(f"Corrupt performance file: {path}, resetting.")
            return {"returns": [], "win_flags": []}

    def record(self, policy_name: str, reward: float, win_flag: int):
        data = self._load(policy_name)
        data["returns"].append(float(reward))
        data["win_flags"].append(int(win_flag))

        out_path = self.perf_root / f"{policy_name}.json"
        out_path.write_text(json.dumps(data, indent=4))

        logger.info(
            f"📊 Perf update for {policy_name}: reward={reward:.4f} win={win_flag}"
        )
