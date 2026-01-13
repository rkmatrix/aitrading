# ai/guardian/kill_switch.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

KILL_SWITCH_FILE = Path("data/runtime/kill_switch.json")


def activate(reason: str = "unknown") -> None:
    """Write kill-switch file. Supervisor will stop restarting Phase 26."""
    data = {"kill": True, "reason": reason}
    KILL_SWITCH_FILE.write_text(json.dumps(data, indent=2))


def deactivate() -> None:
    """Remove kill-switch file."""
    if KILL_SWITCH_FILE.exists():
        KILL_SWITCH_FILE.unlink()


def status() -> Dict[str, Any]:
    """Returns the kill-switch JSON (or {'kill': False} if not present)."""
    if not KILL_SWITCH_FILE.exists():
        return {"kill": False}
    try:
        return json.loads(KILL_SWITCH_FILE.read_text())
    except Exception:
        return {"kill": True, "reason": "corrupt file"}
