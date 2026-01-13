from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AgentState:
    status: str = "ACTIVE"   # ACTIVE | PROBATION | RETIRED
    tier: str = "T0"         # T0..T3
    last_updated: float = 0.0
    notes: str = ""


class AgentGovernor:
    """
    Phase D-8: Agent retirement + promotion + capital tiers.
    """

    def __init__(
        self,
        *,
        state_path: str = "data/governance/agent_governance.json",
        min_trades_for_actions: int = 12,
        retire_trust: float = 0.22,
        probation_trust: float = 0.30,
        promote_t1: float = 0.55,
        promote_t2: float = 0.70,
        promote_t3: float = 0.82,
    ) -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.min_trades_for_actions = min_trades_for_actions
        self.retire_trust = retire_trust
        self.probation_trust = probation_trust
        self.promote_t1 = promote_t1
        self.promote_t2 = promote_t2
        self.promote_t3 = promote_t3

        self.agents: Dict[str, AgentState] = {}
        self._load()

    def get(self, agent_id: str) -> AgentState:
        return self.agents.get(agent_id, AgentState())

    def evaluate(
        self,
        *,
        trust_map: Dict[str, float],
        trades_map: Optional[Dict[str, int]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, AgentState]:
        now = now or time.time()
        trades_map = trades_map or {}

        for agent_id, trust in trust_map.items():
            trades = trades_map.get(agent_id, 0)
            st = self.agents.get(agent_id, AgentState())

            if trades < self.min_trades_for_actions:
                st.status = "ACTIVE"
                st.tier = "T0"
                st.notes = f"warming_up trades={trades}"
            else:
                if trust <= self.retire_trust:
                    st.status = "RETIRED"
                    st.tier = "T0"
                    st.notes = f"retired trust={trust:.3f}"
                elif trust <= self.probation_trust:
                    st.status = "PROBATION"
                    st.tier = "T0"
                    st.notes = f"probation trust={trust:.3f}"
                else:
                    st.status = "ACTIVE"
                    if trust >= self.promote_t3:
                        st.tier = "T3"
                    elif trust >= self.promote_t2:
                        st.tier = "T2"
                    elif trust >= self.promote_t1:
                        st.tier = "T1"
                    else:
                        st.tier = "T0"
                    st.notes = f"active trust={trust:.3f}"

            st.last_updated = now
            self.agents[agent_id] = st

        self._save()
        return self.agents

    def weight_multiplier(self, agent_id: str) -> float:
        st = self.get(agent_id)
        if st.status == "RETIRED":
            return 0.02
        if st.status == "PROBATION":
            return 0.35
        if st.tier == "T3":
            return 1.35
        if st.tier == "T2":
            return 1.20
        if st.tier == "T1":
            return 1.08
        return 1.00

    def _save(self) -> None:
        payload = {
            "ts": time.time(),
            "agents": {k: asdict(v) for k, v in self.agents.items()},
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.agents = {k: AgentState(**v) for k, v in data.get("agents", {}).items()}
        except Exception:
            self.agents = {}
