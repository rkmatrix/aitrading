"""
ai/allocation/multi_symbol_allocator.py
---------------------------------------

Phase 28: Multi-Symbol Allocator

Coordinates per-symbol MicroAllocator decisions into a
portfolio-aware, multi-symbol allocation.

IMPORTANT:
- This class must NOT instantiate itself inside __init__ (that causes recursion).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ai.allocation.micro_allocator import MicroAllocator, MicroAllocDecision

log = logging.getLogger("MultiSymbolAllocator")


@dataclass
class MultiAllocConfig:
    # Max number of symbols we are allowed to be active at once
    max_active_symbols: int = 3

    # Min abs(score) before we even consider a symbol (extra guard)
    min_abs_score: float = 0.05

    # Portfolio-level gross exposure constraint (sum of |weights|)
    portfolio_max_gross_weight: float = 1.00  # 100% of equity

    # Optional per-symbol cap override (otherwise uses MicroAllocator's max_weight)
    per_symbol_max_weight: Optional[float] = None


@dataclass
class MultiSymbolDecision:
    symbol: str
    rank: int
    fused_score: float
    ensemble_score: float
    alloc: MicroAllocDecision


class MultiSymbolAllocator:
    """
    Phase 28 Multi-Symbol Allocator.

    Uses an underlying MicroAllocator instance to size each symbol,
    then applies portfolio-level constraints to pick a subset.
    """

    def __init__(
        self,
        micro_allocator: MicroAllocator,
        cfg: Optional[MultiAllocConfig] = None,
    ) -> None:
        self.log = logging.getLogger("MultiSymbolAllocator")
        self.micro_alloc = micro_allocator

        # Base config (either passed in, or defaults)
        base_cfg = cfg or MultiAllocConfig()

        # Apply env overrides (stable + backwards compatible)
        # Override fields on a NEW config object so the caller's cfg is not mutated.
        self.cfg = MultiAllocConfig(
            max_active_symbols=int(
                os.getenv("PHASE28_MAX_ACTIVE", str(base_cfg.max_active_symbols))
            ),
            min_abs_score=float(
                os.getenv("PHASE28_MIN_ABS_SCORE", str(base_cfg.min_abs_score))
            ),
            portfolio_max_gross_weight=float(
                os.getenv(
                    "PHASE28_MAX_GROSS_WEIGHT",
                    str(base_cfg.portfolio_max_gross_weight),
                )
            ),
            per_symbol_max_weight=base_cfg.per_symbol_max_weight,
        )

        # Optional: allow overriding per-symbol cap via env (empty/None disables)
        env_cap = os.getenv("PHASE28_PER_SYMBOL_MAX_WEIGHT", "").strip()
        if env_cap:
            try:
                self.cfg.per_symbol_max_weight = float(env_cap)
            except Exception:
                self.log.warning(
                    "Invalid PHASE28_PER_SYMBOL_MAX_WEIGHT=%r; ignoring.", env_cap
                )

        self.log.info(
            "MultiSymbolAllocator initialized: max_active=%d, "
            "min_abs_score=%.4f, portfolio_max_gross_weight=%.2f, "
            "per_symbol_max_weight=%s",
            self.cfg.max_active_symbols,
            self.cfg.min_abs_score,
            self.cfg.portfolio_max_gross_weight,
            str(self.cfg.per_symbol_max_weight),
        )

    def allocate(
        self,
        *,
        equity: float,
        positions: Dict[str, Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        volatility_map: Optional[Dict[str, float]] = None,
        mode: str = "PAPER",
        ctx: Optional[Dict[str, Any]] = None,
    ) -> List[MultiSymbolDecision]:
        ctx = ctx or {}
        volatility_map = volatility_map or {}
        mode_u = str(mode).upper()

        if equity <= 0:
            self.log.warning("allocate: equity=%.4f ≤ 0 → no trades.", equity)
            return []

        if not candidates:
            self.log.info("allocate: no candidates → no trades.")
            return []

        # 1) Sort candidates by absolute ensemble score (strongest first)
        scored = []
        for c in candidates:
            sym = c.get("symbol")
            if sym is None:
                continue
            fused = float(c.get("fused_score", 0.0) or 0.0)
            ensemble = float(c.get("ensemble_score", fused) or fused)
            scored.append((sym, fused, ensemble, c))

        if not scored:
            self.log.info("allocate: no valid scored candidates.")
            return []

        scored.sort(key=lambda t: abs(t[2]), reverse=True)

        # 2) Iterate symbols in descending importance, stop at max_active
        decisions: List[MultiSymbolDecision] = []
        total_gross_weight = 0.0

        for rank, (sym, fused_score, ensemble_score, cdict) in enumerate(
            scored, start=1
        ):
            if len(decisions) >= self.cfg.max_active_symbols:
                break

            abs_score = abs(ensemble_score)
            if abs_score < self.cfg.min_abs_score and mode_u != "DEMO":
                self.log.info(
                    "Symbol %s ensemble_score=%.4f below min_abs_score=%.4f; skipping.",
                    sym,
                    ensemble_score,
                    self.cfg.min_abs_score,
                )
                continue

            price = float(cdict.get("price", 0.0) or 0.0)
            if price <= 0:
                self.log.warning(
                    "Symbol %s has invalid price %.4f; skipping.", sym, price
                )
                continue

            pos = positions.get(sym, {"qty": 0.0, "price": price})
            cur_qty = float(pos.get("qty", 0.0) or 0.0)

            vol = float(volatility_map.get(sym, 1.0) or 1.0)

            alloc_decision = self.micro_alloc.compute(
                symbol=sym,
                fused=ensemble_score,
                price=price,
                equity=equity,
                position_qty=cur_qty,
                volatility=vol,
                ctx={"mode": mode_u, **ctx},
            )

            if not alloc_decision.should_trade:
                self.log.info(
                    "MultiSymbolAllocator: no-trade for %s → %s",
                    sym,
                    alloc_decision.clamp_reason,
                )
                continue

            tgt_w = abs(float(alloc_decision.target_weight))

            # Optional per-symbol cap override
            if self.cfg.per_symbol_max_weight is not None:
                if tgt_w > self.cfg.per_symbol_max_weight:
                    self.log.info(
                        "Symbol %s target_weight=%.4f > per_symbol_max_weight=%.4f; clamping.",
                        sym,
                        tgt_w,
                        self.cfg.per_symbol_max_weight,
                    )
                    scale = self.cfg.per_symbol_max_weight / max(tgt_w, 1e-9)
                    alloc_decision.target_weight *= scale
                    alloc_decision.target_notional *= scale
                    alloc_decision.target_qty = int(
                        float(alloc_decision.target_qty) * scale
                    )
                    tgt_w = abs(float(alloc_decision.target_weight))

            # Portfolio-level gross exposure check
            if (total_gross_weight + tgt_w) > self.cfg.portfolio_max_gross_weight:
                self.log.info(
                    "Skipping %s: adding weight=%.4f would exceed portfolio_max_gross_weight=%.4f "
                    "(current total=%.4f).",
                    sym,
                    tgt_w,
                    self.cfg.portfolio_max_gross_weight,
                    total_gross_weight,
                )
                continue

            total_gross_weight += tgt_w

            decisions.append(
                MultiSymbolDecision(
                    symbol=sym,
                    rank=rank,
                    fused_score=float(fused_score),
                    ensemble_score=float(ensemble_score),
                    alloc=alloc_decision,
                )
            )

        self.log.info(
            "MultiSymbolAllocator: selected %d symbols | total_gross_weight=%.4f",
            len(decisions),
            total_gross_weight,
        )

        return decisions
