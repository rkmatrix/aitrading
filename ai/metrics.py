import numpy as np
from pathlib import Path
import json

class Checkpointer:
    def __init__(self, latest="models/ppo_latest.zip", best="models/ppo_best.zip", meta="models/ppo_meta.json"):
        self.latest = Path(latest)
        self.best = Path(best)
        self.meta = Path(meta)
        self.best_score = -1e9
        self._load_meta()

    def _load_meta(self):
        if self.meta.exists():
            try:
                m = json.loads(self.meta.read_text())
                self.best_score = m.get("best_score", self.best_score)
            except Exception:
                pass

    def consider(self, agent, score):
        agent.save(self.latest.as_posix())
        if score > self.best_score:
            agent.save(self.best.as_posix())
            self.best_score = score
            self.meta.write_text(json.dumps({"best_score": self.best_score}))
            print(f"🏆 New best checkpoint saved (score={score:.4f})")
