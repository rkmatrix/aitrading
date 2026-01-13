from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging

from ai.execution.order_schema import Order, OrderResult
from ai.execution.brokers.base import Broker
from ai.execution.brokers.dry import DryRunBroker
from ai.execution.brokers.alpaca import AlpacaBroker

log = logging.getLogger("SmartOrderRouter")

class SmartOrderRouter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.dry_run = bool(cfg.get("dry_run", True))
        self.brokers: list[Broker] = []
        self._init_brokers()

    def _init_brokers(self):
        brokers_cfg = (self.cfg.get("router") or {}).get("brokers", [])
        for b in brokers_cfg:
            if not b.get("enabled", False):
                continue
            name = b.get("name")
            params = b.get("params", {})
            if name == "dry":
                self.brokers.append(DryRunBroker(name="dry", params=params,
                                                 cache_path=(self.cfg.get("account") or {}).get("dry_positions_cache")))
            elif name == "alpaca":
                self.brokers.append(AlpacaBroker(name="alpaca", params=params))
            else:
                log.warning("Unknown broker '%s' – skipping", name)

        if not self.brokers:
            # Always have a dry broker fallback
            self.brokers.append(DryRunBroker())

    def _choose_broker(self) -> Broker:
        # Very simple policy: first broker whose preflight passes (or always dry in dry_run)
        if self.dry_run:
            for b in self.brokers:
                if isinstance(b, DryRunBroker):
                    return b
            return DryRunBroker()
        for b in self.brokers:
            ok, _ = b.preflight()
            if ok:
                return b
        return DryRunBroker()

    def get_positions(self, symbols: List[str]) -> Dict[str, float]:
        # Query chosen broker for positions
        b = self._choose_broker()
        try:
            return b.get_positions(symbols)
        except Exception as e:
            log.error("get_positions failed: %s", e)
            return {s: 0.0 for s in symbols}

    def route(self, order: Order) -> OrderResult:
        b = self._choose_broker()
        if self.dry_run and not isinstance(b, DryRunBroker):
            # Force safety
            b = DryRunBroker()
        try:
            return b.place_order(order)
        except Exception as e:
            return OrderResult(ok=False, broker=b.name, order_id=None, status="error", error=str(e))
