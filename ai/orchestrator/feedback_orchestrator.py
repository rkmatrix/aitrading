# ai/orchestrator/feedback_orchestrator.py
from __future__ import annotations
import json, time, hashlib, logging, subprocess, sys
from pathlib import Path
from typing import Dict, Any, Tuple

def file_fingerprint(p: Path) -> Tuple[int, int, str]:
    if not p.exists(): return (0, 0, "")
    data = p.read_bytes()
    return (int(p.stat().st_mtime), len(data), hashlib.sha1(data).hexdigest())

class PhaseOrchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log = logging.getLogger("Phase45")
        self.state_path = Path(cfg["state"]["path"])
        self.state = {
            "last_run_ts": 0.0,
            "last_feedback_fp": [0, 0, ""],
            "last_selection_fp": [0, 0, ""],
            "last_training_fp": [0, 0, ""],
            "runs": 0
        }
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _should_run_now(self) -> bool:
        min_gap = float(self.cfg["policy"].get("min_gap_minutes", 10)) * 60.0
        trig_every = float(self.cfg["policy"].get("trigger_on_time_minutes", 0)) * 60.0
        now = time.time()
        last = float(self.state.get("last_run_ts", 0.0))
        if min_gap and (now - last < min_gap):
            return False
        if trig_every and (now - last >= trig_every):
            return True
        # otherwise depend on feedback change
        return False

    def _detect_changes(self) -> Dict[str, bool]:
        sig = self.cfg["signals"]
        feedback_fp = file_fingerprint(Path(sig["feedback_csv"]))
        selection_fp = file_fingerprint(Path(sig["selection_csv"]))
        training_fp = file_fingerprint(Path(sig["training_csv"]))

        changed = {
            "feedback": feedback_fp != tuple(self.state.get("last_feedback_fp", (0,0,""))),
            "selection": selection_fp != tuple(self.state.get("last_selection_fp", (0,0,""))),
            "training": training_fp != tuple(self.state.get("last_training_fp", (0,0,"")))
        }
        # update state cache (but not last_run_ts)
        self.state["last_feedback_fp"] = list(feedback_fp)
        self.state["last_selection_fp"] = list(selection_fp)
        self.state["last_training_fp"] = list(training_fp)
        return changed

    def _run_phase(self, module: str, cfg_path: str) -> bool:
        if self.cfg["policy"].get("dry_chain", False):
            self.log.info(f"[DRY] would run: python -m {module} --config {cfg_path}")
            return True
        cmd = [sys.executable, "-m", module, "--config", cfg_path]
        self.log.info("▶ " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            self.log.error(r.stdout)
            self.log.error(r.stderr)
            return False
        self.log.info(r.stdout.strip())
        return True

    def cycle(self) -> str:
        pol = self.cfg["policy"]
        changes = self._detect_changes()
        self.save_state()  # persist fingerprints

        reason = None
        if pol.get("trigger_on_feedback", True) and changes["feedback"]:
            reason = "feedback_changed"
        elif self._should_run_now():
            reason = "time_budget"

        if not reason:
            return "noop"

        self.log.info(f"⏩ Orchestration triggered by: {reason}")

        # 1) Phase 43 — selection & evolution
        for _ in range(int(pol.get("max_phase43_per_cycle", 1))):
            ok = self._run_phase(self.cfg["phase43"]["module"], self.cfg["phase43"]["config"])
            if not ok: break

        # 2) Phase 44 — adaptive training (consume new variants)
        for _ in range(int(pol.get("max_phase44_per_cycle", 1))):
            ok = self._run_phase(self.cfg["phase44"]["module"], self.cfg["phase44"]["config"])
            if not ok: break

        self.state["last_run_ts"] = time.time()
        self.state["runs"] = int(self.state.get("runs", 0)) + 1
        self.save_state()
        return reason
