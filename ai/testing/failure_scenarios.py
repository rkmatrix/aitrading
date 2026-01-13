# ai/testing/failure_scenarios.py

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PatchSpec:
    """
    Describes a single monkeypatch.

    Attributes
    ----------
    target : str
        Import path to patch, e.g. "ai.execution.broker_alpaca_live.AlpacaClient.submit_order".
    mode : str
        One of:
          - "raise" : replace with fn that raises configured exception.
          - "delay" : wrap the function with an artificial time.sleep().
          - "return" : replace with fn that returns a fixed value.
          - "noop" : replace with a fn that does nothing (for fire-and-forget).
    exception : Optional[str]
        Exception class name to raise, e.g. "RuntimeError" (only for mode="raise").
    message : Optional[str]
        Error message for the raised exception (mode="raise").
    delay_seconds : Optional[float]
        Sleep seconds to inject (mode="delay").
    return_value : Optional[Any]
        Fixed value to return (mode="return").
    """

    target: str
    mode: str
    exception: Optional[str] = None
    message: Optional[str] = None
    delay_seconds: Optional[float] = None
    return_value: Optional[Any] = None
    enabled: bool = True


@dataclass
class FailureScenario:
    """
    A single failure scenario configuration.
    """

    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    patches: List[PatchSpec] = field(default_factory=list)
    max_runtime_seconds: Optional[float] = None
    repeat: int = 1

    extra: Dict[str, Any] = field(default_factory=dict)


def _normalize_patch(raw: Dict[str, Any]) -> PatchSpec:
    """Build a PatchSpec from a dict loaded from YAML."""
    return PatchSpec(
        target=raw["target"],
        mode=str(raw.get("mode", "raise")),
        exception=raw.get("exception"),
        message=raw.get("message"),
        delay_seconds=raw.get("delay_seconds"),
        return_value=raw.get("return_value"),
        enabled=bool(raw.get("enabled", True)),
    )


def build_scenarios(cfg: Dict[str, Any]) -> List[FailureScenario]:
    """
    Build a list of FailureScenario from the YAML config dictionary.

    Expected shape:

    failure_harness:
      target:
        module: runner.phase26_realtime_live
        callable: main
      scenarios:
        - name: Broker outage
          description: Simulate Alpaca being unavailable
          tags: [broker, outage]
          max_runtime_seconds: 60
          repeat: 1
          patches:
            - target: ai.execution.broker_alpaca_live.AlpacaClient.submit_order
              mode: raise
              exception: RuntimeError
              message: "Simulated broker outage"
    """
    root = cfg.get("failure_harness", {})
    scenarios_cfg: List[Dict[str, Any]] = root.get("scenarios", [])

    scenarios: List[FailureScenario] = []
    for s in scenarios_cfg:
        patches_raw = s.get("patches", [])
        patches = [_normalize_patch(p) for p in patches_raw if p.get("target")]
        scenario = FailureScenario(
            name=s["name"],
            description=s.get("description", ""),
            tags=list(s.get("tags", [])),
            patches=patches,
            max_runtime_seconds=s.get("max_runtime_seconds"),
            repeat=int(s.get("repeat", 1)),
            extra={k: v for k, v in s.items()
                   if k not in {"name", "description", "tags", "patches", "max_runtime_seconds", "repeat"}},
        )
        scenarios.append(scenario)
    return scenarios
