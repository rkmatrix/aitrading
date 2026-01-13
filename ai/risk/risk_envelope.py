"""
Phase-92.x Risk Envelope Controller
-----------------------------------
Real-time risk engine for AITradeBot.

Features
--------
✓ Equity-based caps (percent-of-equity)
✓ Per-symbol caps
✓ Per-order caps
✓ Total notional exposure caps
✓ Regime scaling (quiet_trend, rangebound, volatile_trend, chaos, extreme_vol, unknown)
✓ Volatility soft/hard caps
✓ Intraday drawdown kill-switch
✓ Leverage constraints
✓ Automatic clamping vs hard-kill when possible
✓ Dynamic Exposure Scaling (Phase 92.2)
✓ Rebalancer-aware logic (Phase 92.3):
      - Always allow risk-reducing trades (closing/trim)
      - Block only risk-increasing trades when caps are exceeded

Compatible with:
    - SmartOrderRouter v4
    - Phase 26 realtime loop (Ultra)
    - Phase 69C/69D configs
    - Phase 92 Multi-Agent Fusion Engine
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable

import yaml

try:
    from tools.telegram_alerts import notify
except Exception:  # optional dependency
    notify = None

try:
    from ai.risk.dynamic_scaler import DynamicExposureScaler
except Exception:
    # Fallback safe stub if module missing
    class DynamicExposureScaler:  # type: ignore
        def compute_scale(self, ctx: Dict[str, Any]) -> float:
            return 1.0


log = logging.getLogger(__name__)


# =====================================================================
# CONFIG DATA CLASSES
# =====================================================================

@dataclass
class RiskLimits:
    # Absolute caps (legacy compatibility)
    max_total_notional: float = 0.0
    max_symbol_notional: float = 0.0
    max_order_notional: float = 0.0

    # Percent-of-equity caps
    max_total_notional_pct: float = 1.0        # allow up to 100% equity gross exposure
    max_symbol_notional_pct: float = 0.30      # 30% of equity per symbol
    max_order_notional_pct: float = 0.15       # 15% equity per order

    # Drawdown + leverage
    max_intraday_drawdown_pct: float = 0.20
    max_leverage: float = 1.0
    allow_short: bool = False

    # Volatility gating
    volatility_soft_cap: float = 2.0
    volatility_hard_cap: float = 4.0

    # Behavior
    soft_clamp: bool = True


@dataclass
class TelegramConfig:
    enabled: bool = False
    channel: str = "guardian"


@dataclass
class RiskEnvelopeConfig:
    mode: str = "PAPER"
    limits: RiskLimits = field(default_factory=RiskLimits)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    regime_scalers: Dict[str, float] = field(default_factory=dict)


# =====================================================================
# RISK CONTROLLER
# =====================================================================

class RiskEnvelopeController:
    """
    PHASE-92.x Real-Time Risk Engine
    """

    def __init__(
        self,
        cfg: Optional[RiskEnvelopeConfig] = None,
        portfolio_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Backward-compatible constructor.

        Supports:
            - New style (Phase 92.x):
                RiskEnvelopeController(cfg=RiskEnvelopeConfig(...), portfolio_provider=...)
            - Older runner style (Phase 69C/26):
                RiskEnvelopeController(config_path="configs/phase69c_risk_envelope.yaml")
        """
        # If no cfg but config_path is given, build cfg from YAML (same logic as from_yaml).
        if cfg is None:
            if config_path is None:
                raise ValueError("RiskEnvelopeController requires either cfg or config_path")

            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            lim = raw.get("limits", {}) or {}

            limits = RiskLimits(
                max_total_notional=float(lim.get("max_total_notional", 0.0)),
                max_symbol_notional=float(lim.get("max_symbol_notional", 0.0)),
                max_order_notional=float(lim.get("max_order_notional", 0.0)),
                max_total_notional_pct=float(lim.get("max_total_notional_pct", 1.0)),
                max_symbol_notional_pct=float(lim.get("max_symbol_notional_pct", 0.30)),
                max_order_notional_pct=float(lim.get("max_order_notional_pct", 0.15)),
                max_intraday_drawdown_pct=float(lim.get("max_intraday_drawdown_pct", 0.20)),
                max_leverage=float(lim.get("max_leverage", 1.0)),
                allow_short=bool(lim.get("allow_short", False)),
                volatility_soft_cap=float(lim.get("volatility_soft_cap", 2.0)),
                volatility_hard_cap=float(lim.get("volatility_hard_cap", 4.0)),
                soft_clamp=bool(lim.get("soft_clamp", True)),
            )

            tel_raw = raw.get("telegram", {}) or {}
            tel = TelegramConfig(
                enabled=bool(tel_raw.get("enabled", False)),
                channel=str(tel_raw.get("channel", "guardian")),
            )

            regime_scalers = raw.get("regime", {}) or {}

            cfg = RiskEnvelopeConfig(
                mode=str(raw.get("mode", "PAPER")),
                limits=limits,
                telegram=tel,
                regime_scalers=regime_scalers,
            )

        # Normal initialization using cfg
        self.cfg = cfg
        self.portfolio_provider = portfolio_provider

        # Regime scaling map
        self._regime_scalers = {
            "quiet_trend": cfg.regime_scalers.get("quiet_trend", 1.0),
            "rangebound": cfg.regime_scalers.get("rangebound", 1.0),
            "volatile_trend": cfg.regime_scalers.get("volatile_trend", 1.0),
            "chaos": cfg.regime_scalers.get("chaos", 1.0),
            "extreme_vol": cfg.regime_scalers.get("extreme_vol", 0.0),
            "unknown": cfg.regime_scalers.get("unknown", 1.0),
        }

        # Phase 92.2: dynamic exposure scaler
        self.dynamic_scaler = DynamicExposureScaler()

    # -----------------------------------------------------------------
    # YAML LOADER
    # -----------------------------------------------------------------
    @classmethod
    def from_yaml(
        cls,
        path: str,
        portfolio_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "RiskEnvelopeController":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        lim = raw.get("limits", {}) or {}

        limits = RiskLimits(
            max_total_notional=float(lim.get("max_total_notional", 0.0)),
            max_symbol_notional=float(lim.get("max_symbol_notional", 0.0)),
            max_order_notional=float(lim.get("max_order_notional", 0.0)),
            max_total_notional_pct=float(lim.get("max_total_notional_pct", 1.0)),
            max_symbol_notional_pct=float(lim.get("max_symbol_notional_pct", 0.30)),
            max_order_notional_pct=float(lim.get("max_order_notional_pct", 0.15)),
            max_intraday_drawdown_pct=float(lim.get("max_intraday_drawdown_pct", 0.20)),
            max_leverage=float(lim.get("max_leverage", 1.0)),
            allow_short=bool(lim.get("allow_short", False)),
            volatility_soft_cap=float(lim.get("volatility_soft_cap", 2.0)),
            volatility_hard_cap=float(lim.get("volatility_hard_cap", 4.0)),
            soft_clamp=bool(lim.get("soft_clamp", True)),
        )

        tel_raw = raw.get("telegram", {}) or {}
        tel = TelegramConfig(
            enabled=bool(tel_raw.get("enabled", False)),
            channel=str(tel_raw.get("channel", "guardian")),
        )

        regime_scalers = raw.get("regime", {}) or {}

        cfg = RiskEnvelopeConfig(
            mode=str(raw.get("mode", "PAPER")),
            limits=limits,
            telegram=tel,
            regime_scalers=regime_scalers,
        )

        return cls(cfg, portfolio_provider)

    # -----------------------------------------------------------------
    # PUBLIC API EXPECTED BY SmartOrderRouter
    # -----------------------------------------------------------------
    def check_and_clamp(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        SmartOrderRouter calls this for every order.

        Returns dict with keys:
            allowed: bool
            order: adjusted_order_dict
            symbol, side, price, equity, drawdown, regime, scale,
            dynamic_scale, hard_kill, clamped, reason, risk_heat,
            volatility, concentration, leverage, is_reducing
        """
        portfolio = self._get_portfolio()

        # Ensure price is present
        order = dict(order)
        if "price" not in order or order["price"] is None:
            inferred = (
                order.get("limit_price")
                or order.get("mid_price")
                or order.get("last_price")
                or 0.0
            )
            order["price"] = float(inferred)

        adjusted, meta = self.evaluate(portfolio, order)
        allowed = (adjusted.get("qty", 0.0) > 0.0) and (not meta.get("hard_kill", False))

        return {
            "allowed": allowed,
            "order": adjusted,
            **meta,
        }

    # -----------------------------------------------------------------
    # INTERNAL: Portfolio + concentration helpers
    # -----------------------------------------------------------------
    def _get_portfolio(self) -> Dict[str, Any]:
        if self.portfolio_provider:
            try:
                return self.portfolio_provider() or {}
            except Exception as e:  # defensive
                log.warning("RiskEnvelope: portfolio_provider failed: %s", e)

        return {
            "equity": 0.0,
            "positions": {},
            "intraday_drawdown_pct": 0.0,
            "regime": "unknown",
        }

    def _calc_concentration(self, positions: Dict[str, Any], equity: float) -> float:
        """
        Max single-symbol notional / equity.
        """
        highest = 0.0
        for _, p in (positions or {}).items():
            qty = abs(float(p.get("qty", 0.0)))
            px = abs(float(p.get("price", 0.0)))
            notional = qty * px
            if notional > highest:
                highest = notional
        return highest / equity if equity > 0 else 0.0

    # -----------------------------------------------------------------
    # CORE EVALUATION
    # -----------------------------------------------------------------
    def evaluate(
        self,
        portfolio: Dict[str, Any],
        order: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate the order against the full risk envelope.

        Returns:
            (adjusted_order, meta)
        """
        sym = order["symbol"]
        side = order["side"]
        qty = float(order["qty"])
        price = float(order.get("price", 0.0))

        equity = float(portfolio.get("equity", 0.0))
        positions = portfolio.get("positions", {}) or {}
        dd = float(portfolio.get("intraday_drawdown_pct", 0.0))
        regime = str(order.get("regime") or portfolio.get("regime") or "unknown")

        limits = self.cfg.limits
        scale = self._regime_scalers.get(regime, self._regime_scalers["unknown"])

        # Existing portfolio totals
        total_notional = 0.0
        for _, p in positions.items():
            pq = float(p.get("qty", 0.0))
            pp = float(p.get("price", price or 0.0))
            total_notional += abs(pq * pp)

        new_notional = abs(qty * price)
        new_total_notional = total_notional + new_notional

        # Position-specific
        pos = positions.get(sym, {})
        pos_qty = float(pos.get("qty", 0.0))
        pos_price = float(pos.get("price", price or 0.0))
        symbol_notional_before = abs(pos_qty * pos_price)
        symbol_notional_after = symbol_notional_before + new_notional

        # Risk-reducing vs increasing (Phase 92.3)
        is_reducing = self._is_risk_reducing_order(side, pos_qty)
        is_increasing = not is_reducing

        # Dynamic exposure scaling context (Phase 92.2)
        vol = float(order.get("volatility", 1.0))
        trend = float(order.get("trend_strength", 0.0))
        ml_var = float(order.get("ml_variance", 0.0))
        conc = self._calc_concentration(positions, equity)
        lev = total_notional / equity if equity > 0 else 1.0

        des_ctx = {
            "equity": equity,
            "volatility": vol,
            "drawdown": dd,
            "trend_strength": trend,
            "ml_variance": ml_var,
            "concentration": conc,
            "leverage": lev,
            "regime": regime,
        }
        dynamic_scale = self.dynamic_scaler.compute_scale(des_ctx)

        meta: Dict[str, Any] = {
            "symbol": sym,
            "side": side,
            "price": price,
            "equity": equity,
            "drawdown": dd,
            "regime": regime,
            "scale": scale,
            "dynamic_scale": dynamic_scale,
            "hard_kill": False,
            "clamped": False,
            "reason": None,
            "risk_heat": None,
            "volatility": vol,
            "concentration": conc,
            "leverage": lev,
            "is_reducing": is_reducing,
        }

        # === ABSOLUTE FAILSAFE: equity <= 0 ===
        if equity <= 0:
            meta["hard_kill"] = True
            meta["reason"] = "equity_le_zero"
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # === DRAWDOWN HARD KILL ===
        if dd >= limits.max_intraday_drawdown_pct:
            # allow risk-reducing even in kill-zone
            if is_reducing:
                meta["reason"] = "drawdown_kill_but_reduce"
                meta["risk_heat"] = 1.0
                return dict(order), meta

            meta["hard_kill"] = True
            meta["reason"] = "drawdown_kill"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # === VOLATILITY GATING ===
        if vol >= limits.volatility_hard_cap:
            if is_reducing:
                meta["reason"] = "volatility_hard_cap_but_reduce"
                meta["risk_heat"] = 1.0
                return dict(order), meta

            meta["hard_kill"] = True
            meta["reason"] = "volatility_hard_cap"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        if vol >= limits.volatility_soft_cap and is_increasing:
            if limits.soft_clamp:
                meta["clamped"] = True
                meta["reason"] = "volatility_soft_cap"
                adj = dict(order)
                adj["qty"] = qty * 0.50
                meta["risk_heat"] = min(1.0, vol / max(1e-9, limits.volatility_hard_cap))
                return adj, meta
            else:
                meta["hard_kill"] = True
                meta["reason"] = "volatility_soft_cap"
                meta["risk_heat"] = 1.0
                return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # === SHORT SELL RESTRICTION ===
        if not limits.allow_short and self._would_increase_short(side, pos_qty, qty):
            meta["hard_kill"] = True
            meta["reason"] = "short_not_allowed"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # ------------------------------------------------------------
        # TOTAL NOTIONAL LIMITS (percent + leverage + absolute)
        # ------------------------------------------------------------
        max_abs = limits.max_total_notional if limits.max_total_notional > 0 else float("inf")
        max_leverage_notional = limits.max_leverage * equity
        max_pct = limits.max_total_notional_pct * equity

        max_total_cap = min(max_abs, max_leverage_notional, max_pct) * scale * dynamic_scale

        if is_increasing and new_total_notional > max_total_cap:
            allowed = max_total_cap - total_notional
            allowed_qty = max(0.0, allowed / price if price > 0 else 0.0)

            if allowed_qty <= 0:
                meta["hard_kill"] = True
                meta["reason"] = "total_notional_cap"
                meta["risk_heat"] = 1.0
                return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

            if limits.soft_clamp:
                meta["clamped"] = True
                meta["reason"] = "total_notional_clamped"
                adj = dict(order)
                adj["qty"] = allowed_qty
                meta["risk_heat"] = min(1.0, new_total_notional / max_total_cap)
                return adj, meta

            meta["hard_kill"] = True
            meta["reason"] = "total_notional_cap"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # ------------------------------------------------------------
        # PER-SYMBOL LIMITS
        # ------------------------------------------------------------
        max_abs_sym = limits.max_symbol_notional if limits.max_symbol_notional > 0 else float("inf")
        max_pct_sym = limits.max_symbol_notional_pct * equity
        max_symbol_cap = min(max_abs_sym, max_pct_sym) * scale * dynamic_scale

        if is_increasing and symbol_notional_after > max_symbol_cap:
            allowed = max_symbol_cap - symbol_notional_before
            allowed_qty = max(0.0, allowed / price if price > 0 else 0.0)

            if allowed_qty <= 0:
                meta["hard_kill"] = True
                meta["reason"] = "symbol_notional_cap"
                meta["risk_heat"] = 1.0
                return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

            if limits.soft_clamp:
                meta["clamped"] = True
                meta["reason"] = "symbol_notional_clamped"
                adj = dict(order)
                adj["qty"] = allowed_qty
                meta["risk_heat"] = symbol_notional_after / max_symbol_cap
                return adj, meta

            meta["hard_kill"] = True
            meta["reason"] = "symbol_notional_cap"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # ------------------------------------------------------------
        # PER-ORDER LIMIT
        # ------------------------------------------------------------
        max_abs_order = limits.max_order_notional if limits.max_order_notional > 0 else float("inf")
        max_pct_order = limits.max_order_notional_pct * equity
        max_order_cap = min(max_abs_order, max_pct_order) * scale * dynamic_scale

        if is_increasing and new_notional > max_order_cap:
            allowed_qty = max_order_cap / price if price > 0 else 0.0

            if allowed_qty <= 0:
                meta["hard_kill"] = True
                meta["reason"] = "order_notional_cap"
                meta["risk_heat"] = 1.0
                return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

            if limits.soft_clamp:
                meta["clamped"] = True
                meta["reason"] = "order_notional_clamped"
                adj = dict(order)
                adj["qty"] = allowed_qty
                meta["risk_heat"] = new_notional / max_order_cap
                return adj, meta

            meta["hard_kill"] = True
            meta["reason"] = "order_notional_cap"
            meta["risk_heat"] = 1.0
            return {"symbol": sym, "side": side, "qty": 0.0, "price": price}, meta

        # ------------------------------------------------------------
        # SUCCESS — allowed order
        # ------------------------------------------------------------
        meta["reason"] = None
        if max_total_cap > 0:
            meta["risk_heat"] = min(1.0, new_total_notional / max_total_cap)
        else:
            meta["risk_heat"] = None

        return dict(order), meta

    # -----------------------------------------------------------------
    # HELPERS (Rebalancer logic)
    # -----------------------------------------------------------------
    @staticmethod
    def _is_risk_reducing_order(side: str, pos_qty: float) -> bool:
        """
        - If we are long (pos_qty > 0), SELL reduces risk.
        - If we are short (pos_qty < 0), BUY reduces risk.
        """
        if pos_qty > 0 and side.upper() == "SELL":
            return True
        if pos_qty < 0 and side.upper() == "BUY":
            return True
        return False

    @staticmethod
    def _would_increase_short(side: str, pos_qty: float, order_qty: float) -> bool:
        """
        Used when allow_short == False to determine whether the order
        would increase net short exposure on a symbol.
        """
        side = side.upper()
        if pos_qty < 0:
            # Already short: SELL increases short
            if side == "SELL" and order_qty > 0:
                return True
        else:
            # Not short: SELL more than current long creates/increases short
            if side == "SELL" and order_qty > pos_qty:
                return True
        return False
