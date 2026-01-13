# ai/reward/pnl_reward_engine.py
"""
PnL-Aware Reward Engine (Phase D-1)
----------------------------------
Turns real (or simulated) trade fills into learning rewards that reflect:
- realized PnL (net of fees/slippage)
- risk normalization (so $10 on tiny risk > $10 on huge risk)
- confidence weighting (so high-confidence wins matter more; high-confidence losses hurt more)
- stability damping (optional, if you pass a stability score)

Designed to be drop-in and safe:
- Works even if you only provide minimal fill fields.
- Persists rewards to JSONL for replay training and audit.
"""

from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FillEvent:
    symbol: str
    side: str                 # "buy" or "sell"
    qty: float
    price: float
    ts: float                 # epoch seconds
    order_id: Optional[str] = None
    broker: Optional[str] = None

    # Optional context fields (good to pass if you have them)
    confidence: Optional[float] = None          # 0..1
    signal_score: Optional[float] = None        # fused signal at decision time
    volatility: Optional[float] = None          # e.g. ATR/price or realized vol (0..)
    stability: Optional[float] = None           # 0..1 (from StabilityLayer)
    regime: Optional[str] = None                # "trend", "range", "risk_off", etc.

    fees: Optional[float] = None                # absolute $ fees for this fill (optional)
    slippage_bps: Optional[float] = None        # slippage in basis points (optional)


@dataclass
class RewardEvent:
    ts: float
    symbol: str
    realized_pnl: float              # $ realized
    realized_pnl_net: float          # $ net (fees+slippage)
    notional: float                  # $ traded notional for the closing portion
    risk_denom: float                # normalization denom used
    reward_raw: float                # before confidence/stability/regime multipliers
    reward: float                    # final reward
    confidence: float
    stability: float
    regime: str
    reason: str                      # "close_long", "close_short", "reduce_long", etc.
    meta: Dict[str, Any]


@dataclass
class PositionState:
    qty: float = 0.0                 # signed: +long, -short
    avg_price: float = 0.0           # average entry price for open qty
    realized_pnl: float = 0.0        # cumulative $ realized
    last_ts: float = 0.0


# -----------------------------
# Engine
# -----------------------------

class PnLRewardEngine:
    """
    Phase D-1 engine.

    Call on_fill() with FillEvent-like dicts.
    When a fill closes/reduces an existing position, a RewardEvent is emitted.
    """

    def __init__(
        self,
        *,
        out_path: str = "data/rewards/pnl_rewards.jsonl",
        state_path: str = "data/rewards/pnl_reward_state.json",
        fee_bps_default: float = 0.0,
        slippage_bps_default: float = 0.0,
        confidence_default: float = 0.50,
        stability_default: float = 1.00,
        regime_default: str = "neutral",
        reward_scale: float = 1.0,
        risk_mode: str = "notional",   # "notional" | "volatility"
        risk_floor: float = 1.0,       # to avoid division by zero
        clip_reward: float = 10.0,     # clip final reward to [-clip, +clip]
    ) -> None:
        self.out_path = Path(out_path)
        self.state_path = Path(state_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.fee_bps_default = float(fee_bps_default)
        self.slippage_bps_default = float(slippage_bps_default)
        self.confidence_default = float(confidence_default)
        self.stability_default = float(stability_default)
        self.regime_default = str(regime_default)
        self.reward_scale = float(reward_scale)
        self.risk_mode = str(risk_mode)
        self.risk_floor = float(risk_floor)
        self.clip_reward = float(clip_reward)

        self.positions: Dict[str, PositionState] = {}
        self._load_state()

        logger.info(
            "PnLRewardEngine initialized | risk_mode=%s out=%s state=%s",
            self.risk_mode, str(self.out_path), str(self.state_path)
        )

    # -----------------------------
    # Public API
    # -----------------------------

    def on_fill(self, fill: Dict[str, Any]) -> Optional[RewardEvent]:
        """
        Process one fill. Returns RewardEvent if realized PnL was generated (close/reduce).
        Otherwise returns None (open/increase position).
        """
        fe = self._normalize_fill(fill)
        sym = fe.symbol.upper()

        pos = self.positions.get(sym, PositionState())
        reward_evt: Optional[RewardEvent] = None

        # Interpret signed delta
        side = fe.side.lower().strip()
        if side not in ("buy", "sell"):
            logger.warning("PnLRewardEngine: unknown side=%r fill=%s", fe.side, fill)
            return None

        delta_qty = fe.qty if side == "buy" else -fe.qty
        if fe.qty <= 0 or fe.price <= 0:
            return None

        # If position is flat, just open/increase
        if abs(pos.qty) < 1e-12:
            pos.qty = delta_qty
            pos.avg_price = fe.price
            pos.last_ts = fe.ts
            self.positions[sym] = pos
            self._save_state_throttled()
            return None

        # If same direction, increase and update avg price
        if (pos.qty > 0 and delta_qty > 0) or (pos.qty < 0 and delta_qty < 0):
            new_qty = pos.qty + delta_qty
            if abs(new_qty) < 1e-12:
                # Shouldn't happen here, but guard anyway
                pos.qty = 0.0
                pos.avg_price = 0.0
            else:
                # Weighted average price
                pos.avg_price = (pos.avg_price * abs(pos.qty) + fe.price * abs(delta_qty)) / abs(new_qty)
                pos.qty = new_qty
            pos.last_ts = fe.ts
            self.positions[sym] = pos
            self._save_state_throttled()
            return None

        # Opposite direction -> reduce/close/flip
        reward_evt = self._handle_reduce_or_close(sym, pos, delta_qty, fe)

        self.positions[sym] = pos
        self._append_reward_if_any(reward_evt)
        self._save_state_throttled()
        return reward_evt

    def get_position(self, symbol: str) -> PositionState:
        return self.positions.get(symbol.upper(), PositionState())

    def flush(self) -> None:
        """Force persist state (and makes sure directories exist)."""
        self._save_state()

    # -----------------------------
    # Internals
    # -----------------------------

    def _handle_reduce_or_close(
        self,
        symbol: str,
        pos: PositionState,
        delta_qty: float,
        fe: FillEvent,
    ) -> Optional[RewardEvent]:
        """
        pos.qty is existing signed position.
        delta_qty is signed change (opposite sign to pos.qty).
        """
        # Determine how much closes existing position
        close_qty = min(abs(delta_qty), abs(pos.qty))
        if close_qty <= 0:
            return None

        # Realized PnL:
        # - For long: selling above avg => +pnl
        # - For short: buying below avg => +pnl
        entry = pos.avg_price
        exitp = fe.price

        if pos.qty > 0:
            # long reduced by sell (delta_qty < 0)
            realized = (exitp - entry) * close_qty
            reason = "close_long" if abs(close_qty - abs(pos.qty)) < 1e-9 else "reduce_long"
        else:
            # short reduced by buy (delta_qty > 0)
            realized = (entry - exitp) * close_qty
            reason = "close_short" if abs(close_qty - abs(pos.qty)) < 1e-9 else "reduce_short"

        # Update position qty after closing portion
        remaining_qty_signed = pos.qty + delta_qty  # delta_qty opposite sign; may flip
        # If flip happened, remaining is new position opened at fill price for leftover qty
        flipped = (pos.qty > 0 and remaining_qty_signed < 0) or (pos.qty < 0 and remaining_qty_signed > 0)

        if abs(remaining_qty_signed) < 1e-12:
            pos.qty = 0.0
            pos.avg_price = 0.0
        elif flipped:
            # leftover opens opposite direction at current fill price
            pos.qty = remaining_qty_signed
            pos.avg_price = fe.price
        else:
            # still same direction, avg_price unchanged
            pos.qty = remaining_qty_signed

        pos.realized_pnl += realized
        pos.last_ts = fe.ts

        # Net PnL adjustments
        notional = close_qty * fe.price
        fees = self._estimate_fees(fe, notional)
        slip = self._estimate_slippage(fe, notional)
        realized_net = realized - fees - slip

        # Risk normalization denom
        risk_denom = self._risk_denom(fe, notional)

        # Reward raw: risk-adjusted net PnL
        reward_raw = (realized_net / max(risk_denom, self.risk_floor)) * self.reward_scale

        # Confidence/stability/regime multipliers
        conf = self._clip01(fe.confidence if fe.confidence is not None else self.confidence_default)
        stab = self._clip01(fe.stability if fe.stability is not None else self.stability_default)
        regime = fe.regime if fe.regime else self.regime_default
        regime_mult = self._regime_multiplier(regime)

        # Make high confidence losses hurt more than low confidence losses,
        # and high confidence wins count more than low confidence wins.
        conf_mult = self._confidence_multiplier(conf, reward_raw)

        # Stability dampens learning when system thinks conditions are unstable.
        stab_mult = 0.5 + 0.5 * stab  # maps 0..1 -> 0.5..1.0

        reward = reward_raw * conf_mult * stab_mult * regime_mult
        reward = self._clip(reward, -self.clip_reward, self.clip_reward)

        meta = {
            "order_id": fe.order_id,
            "broker": fe.broker,
            "entry_price": entry,
            "exit_price": exitp,
            "close_qty": close_qty,
            "pos_qty_after": pos.qty,
            "avg_price_after": pos.avg_price,
            "fees_est": fees,
            "slippage_est": slip,
            "signal_score": fe.signal_score,
            "volatility": fe.volatility,
            "flipped": flipped,
        }

        return RewardEvent(
            ts=fe.ts,
            symbol=symbol,
            realized_pnl=float(realized),
            realized_pnl_net=float(realized_net),
            notional=float(notional),
            risk_denom=float(risk_denom),
            reward_raw=float(reward_raw),
            reward=float(reward),
            confidence=float(conf),
            stability=float(stab),
            regime=str(regime),
            reason=reason,
            meta=meta,
        )

    def _normalize_fill(self, fill: Dict[str, Any]) -> FillEvent:
        # Accept many common schemas
        symbol = (fill.get("symbol") or fill.get("sym") or "").strip()
        side = (fill.get("side") or fill.get("action") or "").strip().lower()

        qty = fill.get("qty", fill.get("quantity", fill.get("filled_qty", 0.0)))
        price = fill.get("price", fill.get("fill_price", fill.get("avg_fill_price", 0.0)))
        ts = fill.get("ts", fill.get("timestamp", fill.get("filled_at", None)))

        # timestamps may be iso string; accept epoch float/int; best-effort for string
        ts_epoch = self._to_epoch(ts)

        return FillEvent(
            symbol=str(symbol),
            side=str(side),
            qty=float(qty) if qty is not None else 0.0,
            price=float(price) if price is not None else 0.0,
            ts=float(ts_epoch),
            order_id=fill.get("order_id") or fill.get("id"),
            broker=fill.get("broker"),
            confidence=fill.get("confidence"),
            signal_score=fill.get("signal_score") or fill.get("fused") or fill.get("score"),
            volatility=fill.get("volatility") or fill.get("atr_norm") or fill.get("vol_norm"),
            stability=fill.get("stability") or fill.get("stability_score"),
            regime=fill.get("regime"),
            fees=fill.get("fees"),
            slippage_bps=fill.get("slippage_bps"),
        )

    def _estimate_fees(self, fe: FillEvent, notional: float) -> float:
        if fe.fees is not None:
            return float(fe.fees)
        # if you don’t have a fee model, keep it conservative and tiny (or 0)
        return (self.fee_bps_default / 10000.0) * float(notional)

    def _estimate_slippage(self, fe: FillEvent, notional: float) -> float:
        bps = fe.slippage_bps if fe.slippage_bps is not None else self.slippage_bps_default
        return (float(bps) / 10000.0) * float(notional)

    def _risk_denom(self, fe: FillEvent, notional: float) -> float:
        """
        risk_mode:
        - "notional": denom = notional (reward ~ pnl / notional)
        - "volatility": denom = notional * max(volatility, small)
          where volatility is ideally ATR/price or realized vol estimate.
        """
        if self.risk_mode == "volatility":
            vol = fe.volatility if fe.volatility is not None else 0.01
            vol = max(float(vol), 1e-4)
            return float(notional) * vol
        return float(notional)

    def _confidence_multiplier(self, conf: float, reward_raw: float) -> float:
        """
        Conf multiplier that:
        - boosts wins when confidence is high
        - penalizes losses harder when confidence is high
        """
        # Map conf 0..1 -> 0.75..1.25
        base = 0.75 + 0.50 * conf
        if reward_raw < 0:
            # high confidence loss => stronger penalty: 1.0..1.5
            return 1.0 + 0.50 * conf
        return base

    def _regime_multiplier(self, regime: str) -> float:
        r = (regime or "neutral").lower().strip()
        # Keep defaults mild; you can tune later.
        if r in ("risk_off", "crash", "panic"):
            return 0.85
        if r in ("trend", "momentum"):
            return 1.05
        if r in ("range", "chop"):
            return 0.95
        return 1.00

    def _append_reward_if_any(self, evt: Optional[RewardEvent]) -> None:
        if evt is None:
            return
        try:
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.exception("PnLRewardEngine: failed writing reward jsonl: %s", e)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            pos_map = data.get("positions", {})
            for sym, p in pos_map.items():
                self.positions[sym] = PositionState(
                    qty=float(p.get("qty", 0.0)),
                    avg_price=float(p.get("avg_price", 0.0)),
                    realized_pnl=float(p.get("realized_pnl", 0.0)),
                    last_ts=float(p.get("last_ts", 0.0)),
                )
        except Exception as e:
            logger.warning("PnLRewardEngine: state load failed, starting fresh: %s", e)

        self._last_state_save = 0.0

    def _save_state(self) -> None:
        try:
            payload = {
                "ts": time.time(),
                "positions": {sym: asdict(p) for sym, p in self.positions.items()},
            }
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._last_state_save = time.time()
        except Exception as e:
            logger.exception("PnLRewardEngine: state save failed: %s", e)

    def _save_state_throttled(self, every_sec: float = 3.0) -> None:
        now = time.time()
        if (now - getattr(self, "_last_state_save", 0.0)) >= every_sec:
            self._save_state()

    @staticmethod
    def _clip01(x: float) -> float:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return 0.5
        return max(0.0, min(1.0, float(x)))

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _to_epoch(ts: Any) -> float:
        if ts is None:
            return time.time()
        if isinstance(ts, (int, float)):
            # if it's huge it's probably ms
            t = float(ts)
            return t / 1000.0 if t > 10_000_000_000 else t
        if isinstance(ts, str):
            s = ts.strip()
            # Try parse ISO-ish without heavy deps; best-effort
            # Accept "2025-12-19T12:34:56Z" or with offset; fallback to now.
            try:
                # Very small parser for "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
                # Note: This ignores timezone offsets; for reward timing this is fine.
                from datetime import datetime
                s2 = s.replace("T", " ").replace("Z", "")
                # trim fractional seconds
                if "." in s2:
                    s2 = s2.split(".", 1)[0]
                dt = datetime.strptime(s2, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
            except Exception:
                return time.time()
        return time.time()
