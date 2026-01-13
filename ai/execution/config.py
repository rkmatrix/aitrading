from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class ExecutionConfig:
    symbols: List[str]
    lob_source: Literal["paper", "replay", "live"] = "paper"
    broker: Literal["paper", "alpaca"] = "paper"
    account_mode: Literal["paper", "live"] = "paper"
    data: Dict[str, Any] = None
    execution: Dict[str, Any] = None
    costs: Dict[str, Any] = None
    rl: Dict[str, Any] = None
    evaluation: Dict[str, Any] = None

    @staticmethod
    def load(path: str | Path) -> "ExecutionConfig":
        d = yaml.safe_load(Path(path).read_text())
        return ExecutionConfig(**d)
