"""
ai/allocators/position_sizer.py
-------------------------------

Phase 34+ PositionSizer with Guardrails (Phase 122.3)

Responsibilities
----------------
- Convert target weights into orders (side + qty) based on prices & equity.
- Respect basic sizing constraints (min notional, max leverage, etc.).
- Run GuardrailRuntimeHandler checks on total notional / open positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ai.safety.auto_guardrails import GuardrailRuntimeHandler, GuardrailDecision

logger = logging.getLogger(__name__)


@dataclass
class PositionSizerConfig:
    max_leverage: float = 2.0
    min_order_notional: float = 50.0
    max_order_notional: float = 25000.0
    round_lot: float = 1.0
    clamp_notional_to_limits: bool = True


class PositionSizer:
    """
    PositionSizer v2
    """

    def __init__(
        self,
        *,
        config_path: str = "configs/phase34_position.yaml",
        guardrails: Optional[GuardrailRuntimeHandler] = None,
    ) -> None:
        self.log = logging.getLogger("PositionSizer")
        self.config_path = Path(config_path)
        self.cfg = self._load_config(self.config_path)
        self.guardrails = guardrails

        self.log.info(
            "PositionSizer initialized (max_leverage=%.2f, min_order_notional=%.2f, max_order_notional=%.2f)",
            self.cfg.max_leverage,
            self.cfg.min_order_notional,
            self.cfg.max_order_notional,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def size_orders(
        self,
        targets: Dict[str, float],
        prices: Dict[str, float],
        account_state: Dict[str, Any],
        current_positions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert target weights into per-symbol orders.

        Returns:
            dict symbol -> {
                "side": "BUY"|"SELL",
                "qty": float,
                "target_weight": float,
                "current_weight": float,
                "notional": float,
            }
        """
        current_positions = current_positions or {}
        equity = float(account_state.get("equity", 0.0)) or 0.0
        if equity <= 0.0:
            self.log.warning("Account equity <= 0; no orders sized.")
            return {}

        orders: Dict[str, Dict[str, Any]] = {}
        total_notional = 0.0

        for sym, target_w in targets.items():
            price = float(prices.get(sym, 0.0) or 0.0)
            if price <= 0.0:
                self.log.debug("Skipping %s: no valid price.", sym)
                continue

            cur_pos = current_positions.get(sym, {})
            cur_w = float(cur_pos.get("weight", 0.0))

            delta_w = target_w - cur_w
            if abs(delta_w) < 1e-4:
                continue

            notional = abs(delta_w) * equity
            if notional < self.cfg.min_order_notional:
                self.log.debug(
                    "Skipping %s: notional %.2f < min_order_notional %.2f",
                    sym,
                    notional,
                    self.cfg.min_order_notional,
                )
                continue

            if self.cfg.clamp_notional_to_limits:
                if notional > self.cfg.max_order_notional:
                    self.log.info(
                        "Clamping %s notional %.2f → %.2f",
                        sym,
                        notional,
                        self.cfg.max_order_notional,
                    )
                    notional = self.cfg.max_order_notional

            qty = notional / price
            if qty <= 0.0:
                continue

            # Round lot
            if self.cfg.round_lot > 0.0:
                qty = (qty // self.cfg.round_lot) * self.cfg.round_lot

            if qty <= 0.0:
                continue

            side = "BUY" if delta_w > 0.0 else "SELL"

            orders[sym] = {
                "side": side,
                "qty": float(qty),
                "target_weight": float(target_w),
                "current_weight": float(cur_w),
                "notional": float(notional),
            }
            total_notional += notional

        # Guardrail check on total notional / open positions
        if self.guardrails is not None and orders:
            open_positions = len(current_positions)
            metrics = {
                "total_notional": float(total_notional),
                "open_positions": float(open_positions),
            }
            ctx = {"symbols": list(orders.keys()), "equity": equity}

            decision: GuardrailDecision = self.guardrails.check(
                event="size_orders",
                metrics=metrics,
                context=ctx,
            )

            if decision.is_blocked:
                self.log.error(
                    "⛔ Guardrails blocked position sizing (total_notional=%.2f, open_pos=%d): %s",
                    total_notional,
                    open_positions,
                    "; ".join(decision.reasons),
                )
                return {}

            if decision.is_clamp:
                self.log.warning(
                    "Guardrails suggested clamp during sizing; metrics=%s",
                    decision.metrics,
                )
                # Optional: we could scale down orders here using decision.metrics

        # Check leverage (gross / equity)
        gross_exposure = sum(abs(o["notional"]) for o in orders.values())
        if gross_exposure > self.cfg.max_leverage * equity:
            scale = (self.cfg.max_leverage * equity / gross_exposure) if gross_exposure > 0 else 1.0
            self.log.info(
                "Scaling orders for leverage control: gross_exposure=%.2f, equity=%.2f, "
                "max_leverage=%.2f → scale=%.3f",
                gross_exposure,
                equity,
                self.cfg.max_leverage,
                scale,
            )
            for sym, o in orders.items():
                o["qty"] = float(o["qty"] * scale)
                o["notional"] = float(o["notional"] * scale)

        return orders

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _load_config(self, path: Path) -> PositionSizerConfig:
        if not path.exists():
            self.log.warning(
                "PositionSizer config %s not found; using defaults.", path
            )
            return PositionSizerConfig()

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            self.log.warning("Failed to load PositionSizer config %s: %s", path, e)
            return PositionSizerConfig()

        cfg = PositionSizerConfig(
            max_leverage=float(raw.get("max_leverage", 2.0)),
            min_order_notional=float(raw.get("min_order_notional", 50.0)),
            max_order_notional=float(raw.get("max_order_notional", 25000.0)),
            round_lot=float(raw.get("round_lot", 1.0)),
            clamp_notional_to_limits=bool(raw.get("clamp_notional_to_limits", True)),
        )
        return cfg
