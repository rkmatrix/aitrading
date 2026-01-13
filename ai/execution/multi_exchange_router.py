from __future__ import annotations
import logging, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import yaml  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    name: str
    kind: str
    priority: int = 1


class BaseBroker:
    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] placing order: %s", self.name, order)
        success = random.random() > 0.1
        if success:
            return {"status": "FILLED", "broker": self.name, "order": order}
        return {"status": "REJECTED", "broker": self.name, "order": order, "error": "sim_fail"}


class PaperBroker(BaseBroker):
    pass


class FeedOnlyBroker(BaseBroker):
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("[%s] is feed_only – cannot place order: %s", self.name, order)
        return {"status": "REJECTED", "broker": self.name, "order": order, "error": "feed_only"}


class MultiExchangeRouter:
    """
    Routes orders across multiple broker endpoints.
    Phase 71–92 compatible.

    Your SmartOrderRouter v4 expects:
        choose_broker(order) -> (broker_name, broker_client, meta)
    """

    def __init__(
        self,
        brokers: Dict[str, BaseBroker],
        default_primary: str,
        fallback_sequence: List[str],
        primary_broker: Optional[Any] = None,
    ) -> None:
        self.brokers = brokers
        self.default_primary = default_primary
        self.fallback_sequence = fallback_sequence
        self.primary_broker = primary_broker   # Alpaca client injected externally

    @classmethod
    def from_yaml(
        cls,
        path: str,
        primary_broker: Optional[Any] = None,
        **kwargs: Any,
    ) -> "MultiExchangeRouter":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        brokers_cfg = raw.get("brokers", []) or []
        brokers: Dict[str, BaseBroker] = {}
        for b in brokers_cfg:
            name = str(b["name"])
            kind = str(b.get("kind", "paper"))

            if kind == "paper":
                brokers[name] = PaperBroker(name, kind)
            elif kind == "feed_only":
                brokers[name] = FeedOnlyBroker(name, kind)
            else:
                brokers[name] = PaperBroker(name, kind)

        routing = raw.get("routing", {}) or {}
        default_primary = str(routing.get("default_primary", "alpaca"))
        fallback_seq = list(routing.get("fallback_sequence", [default_primary]))

        return cls(brokers, default_primary, fallback_seq, primary_broker=primary_broker)

    # ----------------------------------------------------------------------
    # NEW: choose_broker() — REQUIRED BY SmartOrderRouter v4
    # ----------------------------------------------------------------------
    def choose_broker(self, order: Dict[str, Any]):
        """
        Decide which broker will route this order.

        Returns:
            (broker_name, broker_client, route_info)
        """

        # 1) If external Alpaca broker was injected → always prefer it
        if self.primary_broker is not None:
            broker_name = self.default_primary
            return (
                broker_name,
                self.primary_broker,
                {
                    "strategy": "fixed_primary",
                    "rank": 1,
                    "score": 1.0,
                    "all_scores": {broker_name: 1.0},
                },
            )

        # 2) Use fallback sequence with internal simulated brokers
        ranked = []
        for idx, name in enumerate(self.fallback_sequence):
            broker = self.brokers.get(name)
            if broker is None:
                logger.warning("Router: missing broker %s – skipping", name)
                continue

            score = max(0.0, 1.0 - 0.05 * idx)
            ranked.append((name, broker, score, idx + 1))

        if not ranked:
            return (
                None,
                None,
                {
                    "strategy": "empty",
                    "rank": None,
                    "score": 0.0,
                    "all_scores": {},
                },
            )

        # Best broker = rank 1
        name, broker, score, rank = ranked[0]
        all_scores = {n: s for (n, _, s, _) in ranked}

        return (
            name,
            broker,
            {
                "strategy": "fallback_sequence",
                "rank": rank,
                "score": score,
                "all_scores": all_scores,
            },
        )

    # ----------------------------------------------------------------------
    # Legacy adapter: route()
    # ----------------------------------------------------------------------
    def route(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Package broker decision in dict form.
        """
        broker_name, client, info = self.choose_broker(order)
        return {
            "broker": broker_name,
            "client": client,
            **info,
        }

    # ----------------------------------------------------------------------
    # Legacy behavior for older phases
    # ----------------------------------------------------------------------
    def route_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        attempted: List[Dict[str, Any]] = []
        for name in self.fallback_sequence:
            broker = self.brokers.get(name)
            if broker is None:
                logger.warning("Router: broker '%s' missing – skipping", name)
                continue

            res = broker.place_order(order)
            attempted.append(res)
            if res.get("status") == "FILLED":
                return {
                    "status": "OK",
                    "route": name,
                    "result": res,
                    "attempts": attempted,
                }

        return {
            "status": "ERROR",
            "route": None,
            "attempts": attempted,
            "error": "all_brokers_failed",
        }
