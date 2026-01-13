from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


log = logging.getLogger("MicroAllocator")


@dataclass
class MicroAllocConfig:
    profile: str = "moderate"

    # Signal gating
    signal_floor: float = 0.05
    rebalance_band: float = 0.01

    # Weight limits
    max_weight: float = 0.20
    min_weight: float = 0.01

    # DEMO overrides
    demo_max_weight: float = 0.40

    # Volatility scaling
    vol_floor: float = 0.5
    vol_ceiling: float = 2.5

    # Quantity rounding
    qty_rounding: int = 1


@dataclass
class MicroAllocDecision:
    symbol: str
    fused: float
    side: Optional[str]
    target_weight: float
    target_notional: float
    target_qty: int
    current_weight: float
    delta_weight: float
    should_trade: bool
    clamp_reason: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class MicroAllocator:
    """
    Phase 27 MicroAllocator — Moderate profile.

    Phase D-4:
    - Capital scaling via trust × confidence
    - Fully bounded and risk-safe
    """

    def __init__(
        self,
        cfg_path: str = "configs/phase27_micro_alloc.yaml",
        *,
        mode: str = "DEMO",
        env: str = "PAPER_TRADING",
    ) -> None:
        self.mode = mode.upper()
        self.env = env.upper()
        self.cfg_path = cfg_path
        self.cfg = self._load_config(cfg_path)

        self.log = logging.getLogger("MicroAllocator")
        self.log.info(
            "MicroAllocator initialized (profile=%s, mode=%s, env=%s, cfg=%s)",
            self.cfg.profile,
            self.mode,
            self.env,
            cfg_path,
        )

    # --------------------------------------------------
    def _load_config(self, path: str) -> MicroAllocConfig:
        p = Path(path)
        if not p.exists():
            log.warning("MicroAllocator config %s not found. Using defaults.", path)
            return MicroAllocConfig()

        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return MicroAllocConfig(
            profile=raw.get("profile", "moderate"),
            signal_floor=float(raw.get("signal_floor", 0.05)),
            rebalance_band=float(raw.get("rebalance_band", 0.01)),
            max_weight=float(raw.get("max_weight", 0.20)),
            min_weight=float(raw.get("min_weight", 0.01)),
            demo_max_weight=float(raw.get("demo_max_weight", 0.40)),
            vol_floor=float(raw.get("vol_floor", 0.5)),
            vol_ceiling=float(raw.get("vol_ceiling", 2.5)),
            qty_rounding=int(raw.get("qty_rounding", 1)),
        )

    # --------------------------------------------------
    def compute(
        self,
        *,
        symbol: str,
        fused: float,
        price: float,
        equity: float,
        position_qty: float = 0.0,
        volatility: float = 1.0,
        trust: float = 0.20,        # Phase D-4
        confidence: float = 0.50,   # Phase D-4
        ctx: Optional[Dict[str, Any]] = None,
    ) -> MicroAllocDecision:

        ctx = ctx or {}
        mode = str(ctx.get("mode", self.mode)).upper()
        demo = mode == "DEMO"

        # 1) Signal floor (Phase D-5: trust-aware override via ctx)
        signal_floor = float(ctx.get("signal_floor_override", self.cfg.signal_floor))
        if abs(fused) < signal_floor:
            return MicroAllocDecision(
                symbol=symbol,
                fused=fused,
                side=None,
                target_weight=0.0,
                target_notional=0.0,
                target_qty=0,
                current_weight=0.0,
                delta_weight=0.0,
                should_trade=False,
                clamp_reason="signal_below_floor",
                debug={"step": "signal_gate", "signal_floor": signal_floor},
            )


        side = "BUY" if fused > 0 else "SELL"

        # 2) Equity / price sanity
        if equity <= 0 or price <= 0:
            return MicroAllocDecision(
                symbol=symbol,
                fused=fused,
                side=None,
                target_weight=0.0,
                target_notional=0.0,
                target_qty=0,
                current_weight=0.0,
                delta_weight=0.0,
                should_trade=False,
                clamp_reason="invalid_equity_or_price",
            )

        current_notional = position_qty * price
        current_weight = current_notional / equity

        # 3) Cubic signal curve
        base_curve = fused ** 3

        # 4) Volatility scaling
        vol = max(self.cfg.vol_floor, min(volatility, self.cfg.vol_ceiling))
        vol_scale = self.cfg.vol_floor / vol
        curve_vol = base_curve * vol_scale

        # 5) Weight cap
        max_w = self.cfg.demo_max_weight if demo else self.cfg.max_weight
        target_weight = max(-max_w, min(max_w, curve_vol))

        # 6) Enforce min non-zero
        if 0 < abs(target_weight) < self.cfg.min_weight:
            target_weight = math.copysign(self.cfg.min_weight, target_weight)

        # --------------------------------------------------
        # Phase D-4: Capital Scaling (TRUST × CONFIDENCE)
        trust = max(0.0, min(1.0, float(trust)))
        confidence = max(0.0, min(1.0, float(confidence)))

        trust_scale = 0.25 + 1.25 * trust          # 0.25x .. 1.50x
        conf_scale = 0.60 + 0.60 * confidence      # 0.60x .. 1.20x
        scale = trust_scale * conf_scale

        target_weight *= scale
        target_weight = max(-max_w, min(max_w, target_weight))

        self.log.info(
            "📏 D-4 scaling %s | trust=%.3f conf=%.3f scale=%.3f",
            symbol,
            trust,
            confidence,
            scale,
        )

        # 7) Rebalance band (Phase D-5: trust-aware override via ctx)
        rebalance_band = float(ctx.get("rebalance_band_override", self.cfg.rebalance_band))
        delta_weight = target_weight - current_weight
        if abs(delta_weight) < rebalance_band:
            return MicroAllocDecision(
                symbol=symbol,
                fused=fused,
                side=side,
                target_weight=current_weight,
                target_notional=current_weight * equity,
                target_qty=int(position_qty),
                current_weight=current_weight,
                delta_weight=0.0,
                should_trade=False,
                clamp_reason="inside_rebalance_band",
                debug={"step": "rebalance_band", "rebalance_band": rebalance_band, "desired_weight": target_weight},
            )


        # 8) Weight → qty
        target_notional = target_weight * equity
        raw_qty = target_notional / price

        step = max(1, self.cfg.qty_rounding)
        signed_qty = math.copysign(
            math.floor(abs(raw_qty) / step) * step, raw_qty
        )

        if signed_qty == 0:
            return MicroAllocDecision(
                symbol=symbol,
                fused=fused,
                side=side,
                target_weight=current_weight,
                target_notional=current_weight * equity,
                target_qty=int(position_qty),
                current_weight=current_weight,
                delta_weight=delta_weight,
                should_trade=False,
                clamp_reason="qty_rounded_to_zero",
            )

        decision = MicroAllocDecision(
            symbol=symbol,
            fused=fused,
            side=side,
            target_weight=target_weight,
            target_notional=target_notional,
            target_qty=int(signed_qty),
            current_weight=current_weight,
            delta_weight=delta_weight,
            should_trade=True,
            debug={
                "trust": trust,
                "confidence": confidence,
                "scale": scale,
                "volatility": volatility,
                "mode": mode,
            },
        )

        self.log.info(
            "MicroAlloc decision %s: fused=%.4f side=%s cur_w=%.4f → tgt_w=%.4f (Δ=%.4f) → qty=%d",
            symbol,
            fused,
            side,
            current_weight,
            target_weight,
            delta_weight,
            decision.target_qty,
        )

        return decision
