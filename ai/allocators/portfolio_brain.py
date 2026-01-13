"""
ai/allocators/portfolio_brain.py
--------------------------------

Phase 68+ PortfolioBrain with Guardrails (Phase 122.3)

Responsibilities
----------------
- Take in signals, current positions, and account state.
- Produce target weights per symbol.
- Enforce basic portfolio constraints (max weight, max leverage, etc.).
- Run GuardrailRuntimeHandler checks on turnover and exposure.
- Expose a backward-compatible interface for Phase 26:
    - get_realized_vol() → used by Phase 26 Fusion/Stability logic.
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
class PortfolioBrainConfig:
    max_abs_weight: float = 0.25         # 25% max per symbol
    max_gross_leverage: float = 2.0      # max gross exposure / equity
    min_position_notional: float = 100.0
    allow_short: bool = True
    target_volatility: Optional[float] = None  # reserved for future
    # you can extend with more fields as needed


class PortfolioBrain:
    """
    PortfolioBrain v2 (Phase 68+122.3) with Phase 26 compatibility.

    Notes
    -----
    - Works with the newer signature:

          PortfolioBrain(config_path="...", guardrails=...)

      AND with the older Phase 26 style:

          PortfolioBrain(symbols=symbols, cfg_path="configs/phase30_portfolio_brain.yaml")

    - Provides get_realized_vol() used by Phase 26:
        we approximate a "volatility-like" measure from the current
        target weights on each compute_targets() call so that callers
        have a stable, non-crashing API.
    """

    def __init__(
        self,
        *,
        config_path: str = "configs/phase68_portfolio_brain.yaml",
        guardrails: Optional[GuardrailRuntimeHandler] = None,
        # Backwards-compatible args for older callers (Phase 26):
        symbols: Optional[Any] = None,
        cfg_path: Optional[str] = None,
    ) -> None:
        self.log = logging.getLogger("PortfolioBrain")

        # Handle old-style cfg_path argument if provided
        if cfg_path and (
            not config_path
            or config_path == "configs/phase68_portfolio_brain.yaml"
        ):
            self.config_path = Path(cfg_path)
        else:
            self.config_path = Path(config_path)

        self.symbols = symbols or []
        self.cfg = self._load_config(self.config_path)
        self.guardrails = guardrails

        # "Realized vol" proxy used by Phase 26 Fusion/Stability layers.
        # This is a simple risk proxy derived from target weights; it is
        # safe and will never raise AttributeError.
        self._realized_vol: float = 0.0

        self.log.info(
            "PortfolioBrain initialized (config=%s, max_abs_weight=%.3f, max_gross_leverage=%.2f)",
            self.config_path,
            self.cfg.max_abs_weight,
            self.cfg.max_gross_leverage,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_targets(
        self,
        signals: Dict[str, Any],
        current_positions: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute target portfolio weights from signals + current state.

        signals:
            dict symbol -> signal strength (e.g., -1..1 or arbitrary scores)

        current_positions:
            dict symbol -> { 'qty': float, 'market_value': float, 'weight': float, ... }

        account_state:
            dict with keys like 'equity', 'cash', 'pnl', etc.
        """
        if not signals:
            self.log.warning("No signals provided; returning flat targets.")
            return {sym: 0.0 for sym in current_positions.keys()}

        equity = float(account_state.get("equity", 0.0)) or 0.0
        if equity <= 0.0:
            self.log.warning("Account equity <= 0; returning flat targets.")
            return {sym: 0.0 for sym in signals.keys()}

        # Transform signals → raw desired weights (before constraints)
        raw_targets = self._signals_to_raw_weights(signals)

        # Apply per-symbol clamp
        clamped = {
            sym: max(min(w, self.cfg.max_abs_weight), -self.cfg.max_abs_weight)
            for sym, w in raw_targets.items()
        }

        # Normalize to keep gross leverage under control
        gross = sum(abs(w) for w in clamped.values())
        if gross > self.cfg.max_gross_leverage and gross > 0.0:
            scale = self.cfg.max_gross_leverage / gross
            self.log.info(
                "Scaling portfolio gross exposure %.3f → %.3f (scale=%.3f)",
                gross,
                self.cfg.max_gross_leverage,
                scale,
            )
            clamped = {sym: w * scale for sym, w in clamped.items()}
            gross = self.cfg.max_gross_leverage

        # Update our simple "realized vol" proxy so Phase 26 can read it
        self._realized_vol = self._estimate_portfolio_vol_from_weights(clamped)

        # Guardrail checks: turnover, exposure, open positions
        if self.guardrails is not None:
            turnover_pct = self._estimate_turnover_pct(current_positions, clamped)
            open_positions = sum(1 for w in clamped.values() if abs(w) > 1e-6)
            gross_abs = gross

            metrics = {
                "turnover_pct": float(turnover_pct),
                "open_positions": float(open_positions),
                "gross_abs_weight": float(gross_abs),
            }
            ctx = {
                "symbols": list(clamped.keys()),
                "equity": equity,
            }

            decision: GuardrailDecision = self.guardrails.check(
                event="compute_targets",
                metrics=metrics,
                context=ctx,
            )

            if decision.is_blocked:
                self.log.error(
                    "⛔ Guardrails blocked portfolio re-target (turnover=%.3f, open_pos=%d, gross=%.3f): %s",
                    turnover_pct,
                    open_positions,
                    gross_abs,
                    "; ".join(decision.reasons),
                )
                # fallback → keep current weights
                return {
                    sym: float(pos.get("weight", 0.0))
                    for sym, pos in current_positions.items()
                }

            if decision.is_clamp:
                # Optional: you can clamp gross exposure further if decision.metrics specifies.
                self.log.warning(
                    "Guardrails suggested clamp on portfolio; metrics=%s",
                    decision.metrics,
                )

        return clamped

    def get_realized_vol(self) -> float:
        """
        Backward-compatible API for Phase 26.

        Returns a "volatility-like" measure derived from the latest
        target weights. It is NOT a full historical realized volatility,
        but a compact proxy so your Fusion/Stability logic has a numeric
        signal and never crashes.

        If compute_targets() has not been called yet, this may be 0.0.
        """
        try:
            return float(self._realized_vol)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _signals_to_raw_weights(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert arbitrary signals to raw weights.

        Simple default:
            - treat signals as scores,
            - normalize by L1 norm so sum(abs(w)) == 1,
            - then let leverage be controlled later via cfg.max_gross_leverage.
        """
        scores: Dict[str, float] = {}
        for sym, s in signals.items():
            try:
                scores[sym] = float(s)
            except Exception:
                self.log.debug("Non-numeric signal for %s: %r → ignored", sym, s)

        if not scores:
            return {}

        l1 = sum(abs(v) for v in scores.values())
        if l1 <= 0.0:
            return {sym: 0.0 for sym in scores.keys()}

        raw = {sym: v / l1 for sym, v in scores.items()}

        if not self.cfg.allow_short:
            # Shift everything to long-only (crude)
            # negative signals become 0; re-normalize positive
            pos = {sym: max(v, 0.0) for sym, v in raw.items()}
            l1_pos = sum(pos.values())
            if l1_pos > 0.0:
                pos = {sym: v / l1_pos for sym, v in pos.items()}
            raw = pos

        return raw

    def _estimate_turnover_pct(
        self,
        current_positions: Dict[str, Any],
        targets: Dict[str, float],
    ) -> float:
        """
        Rough turnover estimate:
            sum(|target_weight - current_weight|) / 2  (0..1) × 100
        """
        if not current_positions:
            return sum(abs(w) for w in targets.values()) * 100.0

        total = 0.0
        for sym, target_w in targets.items():
            cur_w = 0.0
            pos = current_positions.get(sym)
            if pos is not None:
                try:
                    cur_w = float(pos.get("weight", 0.0))
                except Exception:
                    cur_w = 0.0
            total += abs(target_w - cur_w)

        return (total / 2.0) * 100.0

    def _estimate_portfolio_vol_from_weights(
        self,
        targets: Dict[str, float],
    ) -> float:
        """
        Simple proxy for "portfolio volatility" based solely on weights.

        We use the L2 norm (root sum of squares) of target weights:
            vol_proxy = sqrt(sum(w^2))

        This is always >= 0 and scales with concentration / aggressiveness
        of the portfolio. It is *not* a historical realized volatility, but
        it behaves monotonically with increased risk.
        """
        if not targets:
            return 0.0

        try:
            s = sum(float(w) ** 2 for w in targets.values())
            if s <= 0.0:
                return 0.0
            return s ** 0.5
        except Exception:
            return 0.0

    def _load_config(self, path: Path) -> PortfolioBrainConfig:
        if not path.exists():
            self.log.warning(
                "PortfolioBrain config %s not found; using defaults.", path
            )
            return PortfolioBrainConfig()

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            self.log.warning("Failed to load PortfolioBrain config %s: %s", path, e)
            return PortfolioBrainConfig()

        cfg = PortfolioBrainConfig(
            max_abs_weight=float(raw.get("max_abs_weight", 0.25)),
            max_gross_leverage=float(raw.get("max_gross_leverage", 2.0)),
            min_position_notional=float(raw.get("min_position_notional", 100.0)),
            allow_short=bool(raw.get("allow_short", True)),
            target_volatility=raw.get("target_volatility"),
        )
        return cfg
