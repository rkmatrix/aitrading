
# =====================================================================
# PHASE E — OPERATOR SAFE (MODE LOCKED, NO DEMO OVERRIDE)
#
# FIXES:
# - live_profile.yaml CANNOT override MODE or ENV
# - DEMO broker / SyntheticPriceFeed completely unreachable in PAPER/LIVE
# - Market-closed idle behavior preserved
#
# Generated at 2026-01-03 06:29:39
# =====================================================================


# =====================================================================
# PHASE E WIRED VERSION
# Live-Capital Graduation (Micro Capital → Guarded Scaling)
#
# This phase enables:
# - REAL/PAPER capital with strict graduation rules
# - Kill-switch hardened execution
# - Drawdown-aware capital scaling
# - Trust-gated exposure expansion
#
# Generated at 2025-12-25 14:15:52
# =====================================================================

"""
runner/phase26_realtime_live.py
--------------------------------

Phase 26 – Realtime Execution Loop (Ultra)

This runner wires together:

- Environment loading
- Broker client (Alpaca paper / live)
- SmartOrderRouter (Phase 69C/69D)
- PortfolioBrain (Phase 30)
- EquityCurveAgent (Phase 69E)
- StabilityGuardian (Phase 93 / 111 / 114)
- Policy Fusion (Phase 88 / 89 / 92)
- LiveCapitalGuardian (Phase 111)
- ExecutionHealthMonitor (Phase 114)

Plus:
- Option A: DEMO mode simulated broker (DemoBroker) + fake fills
- Option B: ML signal sensitivity knob
- Option C: Momentum score scaling
- Option D: DEMO-mode PnL curve logging
- Option E: Synthetic price generator for offline testing

"""

from __future__ import annotations

import logging
import sys
import os
import time
import json
import csv
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, Optional
from pathlib import Path
import threading
import traceback

# Add project root to Python path
runner_dir = Path(__file__).parent
project_root = runner_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml
import warnings
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------
from tools.env_loader import ensure_env_loaded
from tools.telegram_alerts import notify
from ai.models.alpha_model import AlphaModel  # NEW
from ai.allocation.micro_allocator import MicroAllocator
from ai.allocation.multi_symbol_allocator import (
    MultiSymbolAllocator,
    MultiAllocConfig,
)
from ai.reward.pnl_reward_engine import PnLRewardEngine
from ai.execution.exit_manager import ExitManager, ExitConfig


# ----------------------------
# Warning hygiene (reduce noise in long-running loops)
# ----------------------------
try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


# ----------------------------
# File logging setup (Phase E)
# ----------------------------
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "phase26_realtime_live.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Avoid duplicate handlers on reload
if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


# ==============================================================
# Phase A — Multi-Symbol Execution Toggle (SAFE DEFAULT: OFF)
# ==============================================================
PHASE28_EXECUTE_ALL = (
    os.getenv("PHASE28_EXECUTE_ALL", "false").strip().lower()
    in ("1", "true", "yes", "on")
)

# tools.error_recorder is optional; provide a safe stub if missing
try:
    from tools.error_recorder import record_error
except ImportError:  # pragma: no cover - fallback stub
    def record_error(*args, **kwargs):
        logging.getLogger("Phase26RealtimeUltra").error(
            "record_error() stub called; tools.error_recorder not available."
        )

# Broker + routing + risk
try:
    from ai.execution.broker_alpaca_live import AlpacaClient
except ImportError:  # pragma: no cover - fallback stub
    class AlpacaClient:  # type: ignore
        """Fallback stub broker when Alpaca client is unavailable.

        Exposes only the methods used in Phase 26:
        - get_account()
        - get_positions()
        - get_last_price(symbol)
        """

        class _Acct:
            def __init__(self, equity: float = 100_000.0, buying_power: float = 200_000.0):
                self.equity = str(equity)
                self.buying_power = str(buying_power)

        class _Pos:
            def __init__(self, symbol: str, qty: float = 0.0, avg_price: float = 0.0):
                self.symbol = symbol
                self.qty = qty
                self.avg_entry_price = avg_price

        def __init__(self, *args, **kwargs) -> None:
            logging.getLogger("Phase26RealtimeUltra").warning(
                "AlpacaClient not available; using in-process stub. "
                "Live/PAPER trading is effectively disabled."
            )
            self._acct = self._Acct()
            self._positions = []

        def get_account(self):
            return self._acct

        def get_positions(self):
            return list(self._positions)

        def get_open_position_map(self):
            return {p.symbol: p for p in self._positions}

        def get_last_price(self, symbol: str):
            # No real price source; caller should fall back to synthetic or skip.
            raise RuntimeError(
                "AlpacaClient stub does not provide real prices; configure real broker."
            )


from ai.execution.smart_order_router import SmartOrderRouter
from ai.risk.risk_envelope import RiskEnvelopeController

# PortfolioBrain is optional; stub out if not present
try:
    from ai.risk.portfolio_brain import PortfolioBrain
except ImportError:  # pragma: no cover - fallback stub
    class PortfolioBrain:
        def __init__(self, *args, **kwargs):
            pass
    
        def get_realized_vol(self) -> float:
            # safe fallback vol
            return 1.0
    
        def get_risk_parity_weights(self):
            return {}

        def get_context(self):
            return {}

        def suggest_allocations(self, *args, **kwargs):
            return {}

# StabilityGuardian optional fallback
try:
    from ai.guardian.stability_guardian import (
        StabilityGuardian,
        StabilityGuardianConfig,
    )
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _sg_dataclass
    from typing import Optional as _sg_Optional, Dict as _sg_Dict, Any as _sg_Any

    @_sg_dataclass
    class StabilityGuardianConfig:  # type: ignore
        max_dd_percent: float = 10.0
        max_intraday_dd_percent: float = 5.0
        min_equity: float = 1000.0
        cooldown_minutes: int = 30

    class StabilityGuardian:  # type: ignore
        def __init__(self, cfg: StabilityGuardianConfig) -> None:
            self.cfg = cfg

        def update(self, equity: float, pnl: float, gross_exposure: float) -> None:
            # No-op in stub
            return

# LiveCapitalGuardian optional fallback
try:
    from ai.risk.live_capital_guardian import (
        LiveCapitalGuardian,
        LiveGuardianConfig as LiveCapitalGuardianConfig,
        LiveGuardianDecision,
    )
    # Use LiveGuardianDecision as LiveCapDecision for compatibility
    LiveCapDecision = LiveGuardianDecision
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _lcg_dataclass
    from typing import Dict as _lcg_Dict, Any as _lcg_Any

    @_lcg_dataclass
    class LiveCapitalGuardianConfig:  # type: ignore
        max_intraday_dd_pct: float = 5.0
        max_daily_dd_pct: float = 10.0
        flatten_all_on_breach: bool = True
        disable_orders_on_breach: bool = True
        cooloff_minutes: int = 60

    @_lcg_dataclass
    class LiveCapDecision:  # type: ignore
        kill_switch_active: bool = False
        flatten_all: bool = False
        should_flatten: bool = False  # Required attribute for compatibility
        disable_new_orders: bool = False
        disable_rl: bool = False  # Also add this for compatibility
        reason: str = ""
        metrics: _lcg_Dict[str, _lcg_Any] = None

    class LiveCapitalGuardian:  # type: ignore
        def __init__(self, cfg: LiveCapitalGuardianConfig) -> None:
            self.cfg = cfg

        def evaluate(self, portfolio_snapshot: dict) -> LiveCapDecision:
            # Always "OK" in stub
            return LiveCapDecision()
        
        def check(self, *, equity: float, positions: list):
            # Stub method to match real implementation signature
            # Returns None to indicate OK (no kill-switch)
            return None

# ExecutionHealthMonitor optional fallback
try:
    from ai.guardian.execution_health import (
        ExecutionHealthMonitor,
        ExecutionHealthConfig,
    )
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _eh_dataclass

    @_eh_dataclass
    class ExecutionHealthConfig:  # type: ignore
        max_error_ticks: int = 5
        max_ticks_since_trade: int = 200
        cooloff_minutes: int = 60

    @_eh_dataclass
    class ExecutionHealthDecision:  # type: ignore
        healthy: bool = True
        reason: str = ""
        flatten_all_positions: bool = False
        raise_kill_switch: bool = False

    class ExecutionHealthMonitor:  # type: ignore
        def __init__(self, cfg: ExecutionHealthConfig) -> None:
            self.cfg = cfg

        def check(self, error_ticks: int, ticks_since_trade: int) -> ExecutionHealthDecision:  # noqa: E501
            return ExecutionHealthDecision()

# FusionBrain optional fallback
try:
    from ai.policy.fusion_brain import FusionBrain, FusionBrainConfig
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _fb_dataclass
    from typing import Dict as _fb_Dict, Any as _fb_Any

    @_fb_dataclass
    class FusionBrainConfig:  # type: ignore
        weights: _fb_Dict[str, float] = None
        alpha_enabled: bool = True

    class FusionBrain:  # type: ignore
        def __init__(self, config: FusionBrainConfig) -> None:
            self.config = config
            self.last_policy_used: str = "StubPolicy"

        def fuse(
            self,
            symbol: str,
            price: float,
            ml_score: float,
            ml_pred: float,
            mom_score: float,
            agg_score: float,
            ctx: _fb_Dict[str, _fb_Any],
        ) -> _fb_Dict[str, _fb_Any]:
            # Simple stub: just pass through ml_score as fused_score
            return {
                "fused_score": ml_score,
                "weights": self.config.weights or {},
                "ppo_val": 0.0,
                "alphazoo_val": 0.0,
            }

from ai.policy.multi_policy_supervisor import MultiPolicySupervisor

# AlphaZooPolicy optional fallback
try:
    from ai.policy.alpha_zoo import AlphaZooPolicy
except ImportError:  # pragma: no cover - fallback stub
    class AlphaZooPolicy:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

# EquityCurveAgent optional fallback
try:
    from ai.policy.equity_agent import EquityCurveAgent, EquityCurveAgentConfig
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _eq_dataclass
    from typing import Dict as _eq_Dict, Any as _eq_Any, List as _eq_List

    @_eq_dataclass
    class EquityCurveAgentConfig:  # type: ignore
        window: int = 50
        max_dd_for_aggressive: float = 0.05
        min_dd_for_defensive: float = 0.15

    class EquityCurveAgent:  # type: ignore
        def __init__(self, cfg: EquityCurveAgentConfig) -> None:
            self.cfg = cfg

        def step(
            self, equity_curve: _eq_List[float], dd: float
        ) -> tuple[float, _eq_Dict[str, _eq_Any]]:
            # Neutral aggression in stub
            return 1.0, {"regime": "neutral", "slope": 0.0}

# MetaStabilityEngine optional fallback
try:
    from ai.policy.meta_stability_engine import MetaStabilityEngine, MetaStabilityConfig
except ImportError:  # pragma: no cover - fallback stub
    from dataclasses import dataclass as _ms_dataclass
    from typing import Dict as _ms_Dict, Any as _ms_Any

    @_ms_dataclass
    class MetaStabilityConfig:  # type: ignore
        lookback: int = 100
        max_drawdown: float = 0.15
        max_vol: float = 0.4
        min_trend_strength: float = 0.1

    @_ms_dataclass
    class MetaDecision:  # type: ignore
        clamp_factor: float = 1.0
        reason: str = ""
        metrics: _ms_Dict[str, _ms_Any] = None

    class MetaStabilityEngine:  # type: ignore
        def __init__(self, cfg: MetaStabilityConfig) -> None:
            self.cfg = cfg

        def update(self, price: float) -> None:
            return

        def evaluate(self) -> MetaDecision:
            return MetaDecision()

from ai.policy.perf_recorder import PerformanceRecorder
from ai.models.alpha_model import AlphaModel, AlphaModelConfig
from ai.risk.symbol_micro_allocator import SymbolMicroAllocator

from ai.signals.regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeThresholds,
)

from ai.risk.symbol_micro_allocator import SymbolMicroAllocator


@dataclass
class SyntheticPriceConfig:
    start_price: float = 100.0
    drift: float = 0.0005
    vol: float = 0.01
    mean_revert_level: float | None = None
    mean_revert_strength: float = 0.05
    seed: int = 42


class SyntheticPriceFeed:
    """Option E – Simple synthetic price generator for offline testing.

    Generates a separate mean-reverting random walk for each symbol so that
    the rest of Phase 26 can run end-to-end without a live data feed.
    """  # noqa: E501

    def __init__(self, cfg: SyntheticPriceConfig) -> None:
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._prices: Dict[str, float] = {}
        self.log = logging.getLogger("SyntheticPriceFeed")

    def _init_symbol(self, symbol: str) -> None:
        if symbol not in self._prices:
            self._prices[symbol] = self.cfg.start_price
            self.log.info(
                "SyntheticPriceFeed: initializing %s at %.2f",
                symbol,
                self.cfg.start_price,
            )

    def next_price(self, symbol: str) -> float:
        self._init_symbol(symbol)
        p = float(self._prices[symbol])

        # Base drift + Gaussian noise
        noise = self._rng.gauss(0.0, self.cfg.vol)
        drift = self.cfg.drift

        # Optional mean reversion around a long-run level
        if self.cfg.mean_revert_level is not None:
            level = float(self.cfg.mean_revert_level)
            # Pull back ~5% of the distance each step (configurable)
            drift += self.cfg.mean_revert_strength * (level - p) / max(p, 1e-6)

        new_p = p * (1.0 + drift + noise)
        new_p = max(new_p, 0.01)
        self._prices[symbol] = new_p
        return float(new_p)


@dataclass
class DemoBrokerPosition:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class DemoAccount:
    equity: float
    buying_power: float


@dataclass
class DemoBrokerSnapshotPos:
    symbol: str
    qty: float
    current_price: float


class DemoBroker:
    """Option A – Minimal in-memory broker for DEMO mode.

    - Simulates fills instantly at the requested price.
    - Tracks cash, positions, and realized PnL.
    - Exposes get_account() / get_positions() so the rest of the loop can
      treat it similarly to AlpacaClient.
    """  # noqa: E501

    def __init__(self, initial_equity: float = 100_000.0, fee_per_trade: float = 0.0) -> None:  # noqa: E501
        self.cash = float(initial_equity)
        self.fee_per_trade = float(fee_per_trade)
        self.positions: Dict[str, DemoBrokerPosition] = {}
        self.realized_pnl: float = 0.0
        self.last_prices: Dict[str, float] = {}
        self.log = logging.getLogger("DemoBroker")

        # Optional: persist fills to disk for Phase C replay pipeline
        ledger_path = os.getenv("PHASE26_DEMO_FILLS_JSONL", "").strip()
        self.ledger_path = ledger_path if ledger_path else None

        if self.ledger_path:
            Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
            self.log.warning("DemoBroker: persisting fills to %s", self.ledger_path)

    # --- Core helpers -------------------------------------------------
    def _get_or_create_pos(self, symbol: str) -> DemoBrokerPosition:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = DemoBrokerPosition(symbol=symbol)
            self.positions[symbol] = pos
        return pos

    def update_mark_prices(self, price_map: Dict[str, float]) -> None:
        """Update last prices for mark-to-market and snapshots."""
        self.last_prices.update({k: float(v) for k, v in price_map.items()})

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        tag: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Simulate a fill and update internal state.

        Returns a dict compatible with SmartOrderRouter-style responses so
        downstream logging / reward code can remain unchanged.
        """  # noqa: E501
        qty = float(qty)
        price = float(price)
        side = side.upper()
        pos = self._get_or_create_pos(symbol)

        signed_qty = qty if side == "BUY" else -qty
        gross = price * qty

        # Simple FIFO-style average price + realized PnL
        pnl = 0.0
        if pos.qty == 0.0 or (pos.qty > 0 and side == "BUY") or (pos.qty < 0 and side == "SELL"):  # noqa: E501
            # Adding to an existing position or opening new in same direction
            total_shares = pos.qty + signed_qty
            if total_shares != 0:
                pos.avg_price = (pos.avg_price * pos.qty + price * signed_qty) / total_shares  # noqa: E501
            pos.qty = total_shares
        else:
            # Closing or flipping direction
            close_qty = min(abs(pos.qty), qty)
            realized = close_qty * (price - pos.avg_price) * (1.0 if pos.qty > 0 else -1.0)  # noqa: E501
            pnl += realized
            pos.qty += signed_qty
            if pos.qty == 0.0:
                pos.avg_price = 0.0
            else:
                pos.avg_price = price

        fee = self.fee_per_trade
        self.cash -= gross if side == "BUY" else -gross
        self.cash -= fee
        self.realized_pnl += pnl

        self.last_prices[symbol] = price

        fill = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "commission": fee,
            "tag": tag,
        }
        # Persist DEMO fill to JSONL for Phase C replay
        if getattr(self, "ledger_path", None):
            try:
                row = {
                    "ts": time.time(),
                    "type": "fill",
                    "fill": fill,
                }
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                self.log.error("DemoBroker: failed to persist fill: %s", e)


        self.log.info(
            "DemoBroker fill: %s %s x %.4f @ %.4f | pnl=%.2f | cash=%.2f",
            side,
            symbol,
            qty,
            price,
            pnl,
            self.cash,
        )

        return {
            "ok": True,
            "id": f"demo-{int(time.time() * 1000)}",
            "raw": fill,
            "fill": fill,
        }

    # --- Snapshot-style helpers --------------------------------------
    def _mark_to_market_unrealized(self) -> float:
        unreal = 0.0
        for sym, pos in self.positions.items():
            px = self.last_prices.get(sym, pos.avg_price)
            unreal += pos.qty * (px - pos.avg_price)
        return unreal

    def mark_to_market(self) -> tuple[float, float, float]:
        """Return (equity, unrealized_pnl, realized_pnl)."""
        unreal = self._mark_to_market_unrealized()
        equity = self.cash + unreal + self.realized_pnl
        return equity, unreal, self.realized_pnl

    def get_account(self) -> DemoAccount:
        equity, unreal, _realized = self.mark_to_market()
        bp = max(self.cash + unreal, 0.0) * 2.0
        return DemoAccount(equity=equity, buying_power=bp)

    def get_positions(self) -> list[DemoBrokerSnapshotPos]:
        out: list[DemoBrokerSnapshotPos] = []
        for sym, pos in self.positions.items():
            if pos.qty == 0.0:
                continue
            px = self.last_prices.get(sym, pos.avg_price)
            out.append(DemoBrokerSnapshotPos(symbol=sym, qty=pos.qty, current_price=px))  # noqa: E501
        return out


class PnLCurveLogger:
    """Option D – DEMO-mode PnL curve logger.

    - Appends (timestamp, equity, cash, unrealized, realized) to CSV.
    - Optionally attempts to export a simple PNG on shutdown.
    """

    def __init__(self, csv_path: str, png_path: str, log_interval_sec: float = 5.0) -> None:  # noqa: E501
        self.csv_path = Path(csv_path)
        self.png_path = Path(png_path)
        self.log_interval_sec = float(log_interval_sec)
        self.last_log_ts: float = 0.0
        self.log = logging.getLogger("PnLCurveLogger")

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "equity", "cash", "unrealized", "realized"])

    def maybe_log(self, equity: float, cash: float, unrealized: float, realized: float) -> None:  # noqa: E501
        now = time.time()
        if now - self.last_log_ts < self.log_interval_sec:
            return
        self.last_log_ts = now

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [now, float(equity), float(cash), float(unrealized), float(realized)]
            )

    def export_png_if_possible(self) -> None:
        """Best-effort PNG export; safe to call even if matplotlib unavailable."""  # noqa: E501
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self.log.warning("PnLCurveLogger: matplotlib not available; skipping PNG export.")  # noqa: E501
            return

        try:
            if not self.csv_path.exists():
                return
            ts: list[float] = []
            eq: list[float] = []
            with self.csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts.append(float(row["ts"]))
                        eq.append(float(row["equity"]))
                    except Exception:
                        continue
            if not ts:
                return
            plt.figure()
            plt.plot(ts, eq)
            plt.xlabel("timestamp")
            plt.ylabel("equity")
            plt.tight_layout()
            self.png_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.png_path)
            plt.close()
        except Exception:
            self.log.exception("PnLCurveLogger: failed to export PNG.")


# --------------------------------------------------------------------
# Logging Config
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("Phase26RealtimeUltra")


# ====================================================================
# Tick Executor (Adaptive Auto-scaling harness)
# ====================================================================
class AdaptiveAutoScalingExecutor:
    """
    Simple utility to run a periodic tick function with adaptive sleep,
    while allowing an external 'running' flag to terminate the loop.
    """

    def __init__(
        self,
        base_interval_sec: float = 5.0,
        min_interval_sec: float = 1.0,
        max_interval_sec: float = 60.0,
    ) -> None:
        self.base_interval_sec = base_interval_sec
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec

    def run(
        self,
        tick_fn,
        state_fn,
        running_flag_fn,
    ) -> None:
        interval = self.base_interval_sec
        tick_idx = 0
        while running_flag_fn():
            start = time.time()

            try:
                state = state_fn()
            except Exception as e:
                log.error("autoscaler state_fn failed: %s", e)
                state = {}

            vol = float(state.get("volatility", 1.0) or 1.0)
            interval = np.clip(
                self.base_interval_sec / max(vol, 0.1),
                self.min_interval_sec,
                self.max_interval_sec,
            )

            ctx = {
                "tick_idx": tick_idx,
                "interval_sec": interval,
                "state": state,
            }

            try:
                tick_fn(ctx)
            except Exception as e:
                log.error("autoscaler tick_fn error: %s", e, exc_info=True)
                record_error("phase26_tick", "exception", str(e))

            elapsed = time.time() - start
            sleep_for = max(interval - elapsed, 0.0)
            time.sleep(sleep_for)
            tick_idx += 1


class RealTimeExecutionLoop:
    """
    Phase 26 Realtime Execution Loop (Ultra).

    This class wires together:
      - Broker + SmartOrderRouter
      - FusionBrain + MultiPolicySupervisor
      - RiskEnvelope + PortfolioBrain
      - StabilityGuardian + LiveCapitalGuardian + ExecutionHealthMonitor
      - EquityCurveAgent for dynamic aggression
      - MetaStabilityEngine (Phase 123)
      - AlphaModel + AlphaZoo policies (Phase 88 / 89 / 92)

    And adds:
      - Option A: DEMO mode simulated broker (DemoBroker) + fake fills
      - Option B: ML signal sensitivity knob
      - Option C: Momentum score scaling
      - Option D: DEMO-mode PnL curve logging
      - Option E: Synthetic price generator for offline testing
    """

    def __init__(self) -> None:
        # -------------------------------------------------------------
        # Basic config
        # -------------------------------------------------------------
        self.mode = os.getenv("MODE", "PAPER").upper()
        self.env = os.getenv("ENV", "PAPER_TRADING").upper()

        # Option B/C – signal sensitivity knobs (env-configurable)
        try:
            self._ml_sensitivity = float(os.getenv("PHASE26_ML_SENSITIVITY", "1.5"))
        except ValueError:
            self._ml_sensitivity = 1.5
        try:
            self._mom_scale = float(os.getenv("PHASE26_MOMENTUM_SCALE", "1.2"))
        except ValueError:
            self._mom_scale = 1.2

        # Option E – synthetic price feed toggle (configured via env)
        synth_flag = os.getenv("PHASE26_USE_SYNTHETIC_PRICES", "").strip().lower()
        self.synthetic_prices_enabled = synth_flag in ("1", "true", "yes", "on")
        self.synthetic_feed: SyntheticPriceFeed | None = None
        # Phase E hard-lock: synthetic prices are DEMO-only.
        if self.mode.upper() != "DEMO":
            if self.synthetic_prices_enabled:
                log.warning(
                    "Synthetic prices requested but MODE=%s; disabling synthetic feed in PAPER/LIVE.",
                    self.mode,
                )
            self.synthetic_prices_enabled = False
        if self.synthetic_prices_enabled:
            try:
                mean_level_str = os.getenv("PHASE26_SYNTH_MEAN_REVERT_LEVEL", "").strip()  # noqa: E501
                mean_level = float(mean_level_str) if mean_level_str else None
            except ValueError:
                mean_level = None
            cfg = SyntheticPriceConfig(
                start_price=float(os.getenv("PHASE26_SYNTH_START_PRICE", "100.0")),
                drift=float(os.getenv("PHASE26_SYNTH_DRIFT", "0.0005")),
                vol=float(os.getenv("PHASE26_SYNTH_VOL", "0.01")),
                mean_revert_level=mean_level,
                mean_revert_strength=float(
                    os.getenv("PHASE26_SYNTH_MEAN_REVERT_STRENGTH", "0.05")
                ),
                seed=int(os.getenv("PHASE26_SYNTH_SEED", "42")),
            )
            self.synthetic_feed = SyntheticPriceFeed(cfg)

        # Kill-switch flag path (Phase 93 / 111 / 114)
        self.kill_flag_path = Path(
            os.getenv("TRADING_KILL_FLAG", "data/runtime/trading_disabled.flag")
        )
        # Track whether a kill flag was active at startup or earlier tick
        self._kill_flag_was_active = self.kill_flag_path.exists()

        # -------------------------------------------------------------
        # Phase 115 – Live Profile wiring (symbol universe, etc.)
        # -------------------------------------------------------------
        live_profile_path = Path("configs/live_profile.yaml")
        if live_profile_path.exists():
            try:
                with live_profile_path.open("r", encoding="utf-8") as f:
                    lp_raw = yaml.safe_load(f) or {}
                lp = lp_raw.get("live_profile", lp_raw) or {}

                profile_mode = str(lp.get("mode", self.mode)).upper()
                profile_env = str(lp.get("env", self.env)).upper()
                # Phase E safety: MODE/ENV are locked to .env (operator intent).
                # live_profile.yaml may override symbols & tuning, but never MODE/ENV.
                if profile_mode != self.mode:
                    log.warning(
                        "Live profile requested MODE override (%s → %s) but MODE is locked; ignoring.",
                        self.mode,
                        profile_mode,
                    )
                if profile_env != self.env:
                    log.warning(
                        "Live profile requested ENV override (%s → %s) but ENV is locked; ignoring.",
                        self.env,
                        profile_env,
                    )

                self.symbols = list(lp.get("symbols", ["AAPL", "MSFT", "TSLA"]))
                log.info("Live profile symbols: %s", self.symbols)
            except Exception as e:
                log.error("Failed to load live_profile.yaml: %s", e)
                self.symbols = ["AAPL", "MSFT", "TSLA"]
        else:
            self.symbols = ["AAPL", "MSFT", "TSLA"]

        # -------------------------------------------------------------
        # Guardrails: we always construct them, but can be disabled
        # -------------------------------------------------------------
        guardrails_cfg_path = Path("configs/guardrails.yaml")
        self.guardrails_enabled = False
        self.guardrails = {}
        self.guardrails_portfolio = {}

        if guardrails_cfg_path.exists():
            try:
                with guardrails_cfg_path.open("r", encoding="utf-8") as f:
                    self.guardrails = yaml.safe_load(f) or {}
                self.guardrails_enabled = bool(
                    self.guardrails.get("enabled", True)
                )
                self.guardrails_portfolio = self.guardrails.get(
                    "portfolio_limits", {}
                ) or {}
                log.info(
                    "Guardrails loaded (enabled=%s)",
                    self.guardrails_enabled,
                )
            except Exception as e:
                log.error("Failed to load guardrails.yaml: %s", e)
        else:
            log.warning("No guardrails.yaml found; running without guardrails.")
            self.guardrails_enabled = False

        # -------------------------------------------------------------
        # StabilityGuardian (Phase 93 / 111 / 114)
        # -------------------------------------------------------------
        stability_cfg = StabilityGuardianConfig(
            max_dd_percent=float(os.getenv("STAB_MAX_DD_PERCENT", "10.0")),
            max_intraday_dd_percent=float(
                os.getenv("STAB_MAX_INTRADAY_DD_PERCENT", "5.0")
            ),
            min_equity=float(os.getenv("STAB_MIN_EQUITY", "1000.0")),
            cooldown_minutes=int(os.getenv("STAB_COOLDOWN_MINUTES", "30")),
        )
        self.stability_guardian = StabilityGuardian(stability_cfg)

        # -------------------------------------------------------------
        # LiveCapitalGuardian (Phase 111)
        # -------------------------------------------------------------
        # The real LiveCapitalGuardian expects a dict config, not a dataclass
        # Check if we imported the real implementation (it has 'check' method)
        if hasattr(LiveCapitalGuardian, 'check'):
            # Real implementation - use dict config
            lcg_cfg_dict = {
                "max_intraday_loss_pct": float(
                    os.getenv("LCG_MAX_INTRADAY_DD_PCT", "5.0")
                ),
                "max_position_exposure_pct": float(os.getenv("LCG_MAX_POSITION_EXPOSURE_PCT", "25.0")),
                "flag_path": os.getenv("LCG_FLAG_PATH", "data/runtime/trading_disabled.flag"),
                "auto_reset_minutes": int(os.getenv("LCG_COOLOFF_MINUTES", "60")),
                "startup_grace_seconds": int(os.getenv("LCG_STARTUP_GRACE_SECONDS", "120")),
            }
            self.live_cap_guardian = LiveCapitalGuardian(lcg_cfg_dict)
        else:
            # Stub implementation - use dataclass config
            lcg_cfg = LiveCapitalGuardianConfig(
                max_intraday_dd_pct=float(
                    os.getenv("LCG_MAX_INTRADAY_DD_PCT", "5.0")
                ),
                max_daily_dd_pct=float(os.getenv("LCG_MAX_DAILY_DD_PCT", "10.0")),
                flatten_all_on_breach=True,
                disable_orders_on_breach=True,
                cooloff_minutes=int(os.getenv("LCG_COOLOFF_MINUTES", "60")),
            )
            self.live_cap_guardian = LiveCapitalGuardian(lcg_cfg)

        self.guardian_enabled = self.guardrails_enabled

        # -------------------------------------------------------------
        # ExecutionHealthMonitor (Phase 114)
        # -------------------------------------------------------------
        eh_cfg = ExecutionHealthConfig(
            max_error_ticks=int(os.getenv("EH_MAX_ERROR_TICKS", "5")),
            max_ticks_since_trade=int(
                os.getenv("EH_MAX_TICKS_SINCE_TRADE", "200")
            ),
            cooloff_minutes=int(os.getenv("EH_COOLOFF_MINUTES", "60")),
        )
        self.health_monitor = ExecutionHealthMonitor(eh_cfg)
        self.health_monitor_enabled = True

        # -------------------------------------------------------------
        # Exit Manager (Phase 1: Advanced Exit Management)
        # -------------------------------------------------------------
        exit_config_path = Path("configs/exit_strategy.yaml")
        try:
            self.exit_manager = ExitManager(config_path=str(exit_config_path))
            log.info("✅ Exit Manager initialized")
        except Exception as e:
            log.warning("Failed to initialize Exit Manager: %s, using defaults", e)
            self.exit_manager = ExitManager()
        
        # Track last known positions to detect new entries
        self._last_positions: Dict[str, float] = {}
        
        # -------------------------------------------------------------
        # Trade Quality Filter (Phase 2: Trade Quality Filtering)
        # -------------------------------------------------------------
        quality_config_path = Path("configs/trade_quality.yaml")
        try:
            self.trade_quality_filter = TradeQualityFilter(config_path=str(quality_config_path))
            log.info("✅ Trade Quality Filter initialized")
        except Exception as e:
            log.warning("Failed to initialize Trade Quality Filter: %s, using defaults", e)
            self.trade_quality_filter = TradeQualityFilter()
        
        # -------------------------------------------------------------
        # Correlation Manager (Phase 3: Portfolio Correlation Management)
        # -------------------------------------------------------------
        correlation_config_path = Path("configs/correlation_limits.yaml")
        try:
            self.correlation_manager = CorrelationManager(config_path=str(correlation_config_path))
            log.info("✅ Correlation Manager initialized")
        except Exception as e:
            log.warning("Failed to initialize Correlation Manager: %s, using defaults", e)
            self.correlation_manager = CorrelationManager()
        
        # -------------------------------------------------------------
        # Adaptive Sizer (Phase 4: Adaptive Position Sizing)
        # -------------------------------------------------------------
        try:
            self.adaptive_sizer = AdaptiveSizer()
            log.info("✅ Adaptive Sizer initialized")
        except Exception as e:
            log.warning("Failed to initialize Adaptive Sizer: %s, using defaults", e)
            self.adaptive_sizer = AdaptiveSizer()

        # -------------------------------------------------------------
        # Broker client (AlpacaClientAdapter wrapper)
        # -------------------------------------------------------------
        self.broker = AlpacaClient()
        acct = self.broker.get_account()
        # AlpacaClient.get_account() returns a dict, not an object
        equity_str = str(acct.get("equity", "0") if isinstance(acct, dict) else getattr(acct, "equity", "0"))
        bp_str = str(acct.get("buying_power", "0") if isinstance(acct, dict) else getattr(acct, "buying_power", "0"))
        log.info(
            "📈 Alpaca equity = $%s | BP = $%s",
            equity_str,
            bp_str,
        )
        # Cache last known good equity (protect against API hiccups)
        equity_val = acct.get("equity", 0.0) if isinstance(acct, dict) else getattr(acct, "equity", 0.0)
        self._last_good_equity = float(equity_val or 0.0)

        
        self._equity_invalid_last_log_ts = 0.0
        self._equity_invalid_last_price_log_ts = 0.0
# DEMO mode flag (used across options A/D)
        self.demo_mode = self.mode.upper() == "DEMO"

        # Option A/D – DemoBroker (simulated fills) + PnL curve logger
        self.demo_broker: DemoBroker | None = None
        self.pnl_logger: PnLCurveLogger | None = None
        if self.demo_mode:
            # DEMO equity seeding:
            # 1) explicit env override (recommended)
            # 2) fallback to broker equity
            # 3) final fallback to 100k
            seed_env = os.getenv("PHASE26_DEMO_INITIAL_EQUITY", "").strip()
            initial_equity = 0.0
        
            if seed_env:
                try:
                    initial_equity = float(seed_env)
                except Exception:
                    initial_equity = 0.0
        
            if initial_equity <= 0.0:
                try:
                    # AlpacaClient.get_account() returns a dict
                    if isinstance(acct, dict):
                        initial_equity = float(acct.get("equity", 0.0) or 0.0)
                    else:
                        initial_equity = float(getattr(acct, "equity", 0.0) or 0.0)
                except Exception:
                    initial_equity = 0.0
        
            if initial_equity <= 0.0:
                initial_equity = 100_000.0
        
            self.demo_broker = DemoBroker(
                initial_equity=initial_equity,
                fee_per_trade=float(os.getenv("PHASE26_DEMO_FEE_PER_TRADE", "0.0")),
            )
        
            # Sanity log so you can see it clearly in startup logs
            log.warning("DEMO broker seeded equity = %.2f", float(initial_equity))
        
            self.pnl_logger = PnLCurveLogger(
                csv_path=os.getenv(
                    "PHASE26_DEMO_PNL_CSV", "data/reports/demo_pnl_phase26.csv"
                ),
                png_path=os.getenv(
                    "PHASE26_DEMO_PNL_PNG", "data/reports/demo_pnl_phase26.png"
                ),
                log_interval_sec=float(
                    os.getenv("PHASE26_DEMO_PNL_INTERVAL_SEC", "5.0")
                ),
            )
            log.warning(
                "Phase 26 DEMO MODE: using DemoBroker (simulated fills) and PnL curve logging.",
            )

            self.pnl_logger = PnLCurveLogger(
                csv_path=os.getenv(
                    "PHASE26_DEMO_PNL_CSV", "data/reports/demo_pnl_phase26.csv"
                ),
                png_path=os.getenv(
                    "PHASE26_DEMO_PNL_PNG", "data/reports/demo_pnl_phase26.png"
                ),
                log_interval_sec=float(
                    os.getenv("PHASE26_DEMO_PNL_INTERVAL_SEC", "5.0")
                ),
            )
            log.warning(
                "Phase 26 DEMO MODE: using DemoBroker (simulated fills) and PnL curve logging.",  # noqa: E501
            )

        # -------------------------------------------------------------
        # PortfolioBrain for volatility/regime context
        # -------------------------------------------------------------
        self.pbrain = PortfolioBrain(
            config_path="configs/phase30_portfolio_brain.yaml",
            guardrails=self.guardrails_portfolio,
        )
        log.info("🧠 PortfolioBrain initialized")

        # -------------------------------------------------------------
        # RiskEnvelopeController (Phase 69C)
        # -------------------------------------------------------------
        try:
            self.risk_envelope = RiskEnvelopeController(
                config_path="configs/phase69c_risk_envelope.yaml",
                portfolio_provider=self._portfolio_snapshot,
            )
        except TypeError:
            # Older version fallback
            self.risk_envelope = RiskEnvelopeController()


        # -------------------------------------------------------------
        # SmartOrderRouter (Phase 69C / 69D)
        # -------------------------------------------------------------
        self.router = SmartOrderRouter(
            risk_cfg_path="configs/phase69c_risk_envelope.yaml",
            multix_cfg_path="configs/phase69d_multix.yaml",
            portfolio_provider=self._portfolio_snapshot,
            primary_broker=self.broker,
        )

        # -------------------------------------------------------------
        # FusionBrain / MultiPolicySupervisor / AlphaZoo / AlphaModel
        # -------------------------------------------------------------
        fb_cfg = FusionBrainConfig(
            weights={
                "ml": 0.4,
                "mom": 0.3,
                "agg": 0.1,
                "ppo": 0.1,
                "alphazoo": 0.1,
            },
            alpha_enabled=True,
        )
        self.fusion_brain = FusionBrain(config=fb_cfg)

        self.policy_supervisor = MultiPolicySupervisor()
        self.alpha_zoo_policy = AlphaZooPolicy()

        # -------------------------------------------------------------
        # AlphaModel (Phase 101 — trained ML predictor)
        # -------------------------------------------------------------
        self.alpha_model_cfg = AlphaModelConfig(
            # Updated to correct Phase 101 training output
            model_path="models/alpha/phase101_alpha_model.joblib",
        )
        
        try:
            self.alpha_model = AlphaModel(self.alpha_model_cfg)
            self._alpha_enabled = True
            log.info(
                "AlphaModel loaded for Phase 26 fusion → %s",
                self.alpha_model_cfg.model_path
            )
        except Exception as e:
            self.alpha_model = None
            self._alpha_enabled = False
            log.warning("AlphaModel not available (%s). ML disabled for this session.", e)

        # -------------------------------------------------------------
        # Phase B — Hot-swap (safe model reload without stopping loop)
        # -------------------------------------------------------------
        self.hotswap_enabled = os.getenv("PHASE26_HOTSWAP_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        self.hotswap_check_sec = float(os.getenv("PHASE26_HOTSWAP_CHECK_SEC", "10.0"))
        self.hotswap_last_check_ts = 0.0
        self._hotswap_lock = threading.RLock()
        
        # Track mtimes so we only reload on change
        self._alpha_model_mtime = 0.0
        try:
            if self.alpha_model_cfg and self.alpha_model_cfg.model_path:
                p = Path(self.alpha_model_cfg.model_path)
                if p.exists():
                    self._alpha_model_mtime = p.stat().st_mtime
        except Exception:
            pass
        
        # Optional: policy hot-swap (best-effort). You can add paths later.
        # Comma-separated file paths to watch (zip/joblib/etc). Safe no-op if empty.
        self.hotswap_policy_paths = [
            x.strip()
            for x in os.getenv("PHASE26_HOTSWAP_POLICY_PATHS", "").split(",")
            if x.strip()
        ]
        self._policy_mtimes: Dict[str, float] = {}
        for path in self.hotswap_policy_paths:
            try:
                pp = Path(path)
                self._policy_mtimes[path] = pp.stat().st_mtime if pp.exists() else 0.0
            except Exception:
                self._policy_mtimes[path] = 0.0
        
        log.info(
            "Phase B HotSwap: enabled=%s check_sec=%.1f alpha_path=%s watched_policies=%d",
            self.hotswap_enabled,
            self.hotswap_check_sec,
            getattr(self.alpha_model_cfg, "model_path", None),
            len(self.hotswap_policy_paths),
        )        # -------------------------------------------------------------
        # Cache last known good equity (protect against API hiccups)
        # -------------------------------------------------------------
        # Seeded from broker.get_account() above.
        # CRITICAL: Do NOT reset to 0 in PAPER/LIVE, or we will permanently
        # disable sizing/trading and spam "equity invalid" logs.
        if not np.isfinite(getattr(self, "_last_good_equity", 0.0)) or self._last_good_equity < 0:
            self._last_good_equity = 0.0

        
        
        # -------------------------------------------------------------
        # Phase 27 MicroAllocator (per-symbol sizing)
        # -------------------------------------------------------------
        from ai.allocation.micro_allocator import MicroAllocator
        
        self.micro_alloc = MicroAllocator(
            cfg_path="configs/phase27_micro_alloc.yaml",
            mode=self.mode,
            env=self.env,
        )
        
        
        # -------------------------------------------------------------
        # Phase 28 Multi-Symbol Allocator (portfolio-aware sizing)
        # -------------------------------------------------------------
        from ai.allocation.multi_symbol_allocator import (
            MultiSymbolAllocator,
            MultiAllocConfig,
        )
        
        self.multi_alloc = MultiSymbolAllocator(
            micro_allocator=self.micro_alloc,
            cfg=MultiAllocConfig(
                max_active_symbols=int(os.getenv("PHASE28_MAX_ACTIVE", "3")),
                min_abs_score=float(os.getenv("PHASE28_MIN_ABS_SCORE", "0.05")),
                portfolio_max_gross_weight=float(
                    os.getenv("PHASE28_MAX_GROSS_WEIGHT", "1.0")
                ),
                per_symbol_max_weight=None,
            ),
        )


        # -------------------------------------------------------------
        # EquityCurveAgent (Phase 69E)
        # -------------------------------------------------------------
        eq_cfg = EquityCurveAgentConfig(
            window=50,
            max_dd_for_aggressive=0.05,
            min_dd_for_defensive=0.15,
        )
        self.equity_agent = EquityCurveAgent(eq_cfg)
        self._equity_curve: deque[float] = deque(maxlen=eq_cfg.window)
        self._aggression_factor = 1.0

        # -------------------------------------------------------------
        # MetaStabilityEngine (Phase 123)
        # -------------------------------------------------------------
        me_cfg = MetaStabilityConfig(
            lookback=100,
            max_drawdown=0.15,
            max_vol=0.4,
            min_trend_strength=0.1,
        )
        self.meta_engine = MetaStabilityEngine(me_cfg)

        # -------------------------------------------------------------
        # RegimeDetector (Phase 120)
        # -------------------------------------------------------------
        thresholds = RegimeThresholds(
            bull=0.02,
            bear=-0.02,
            high_vol=0.03,
        )
        rd_cfg = RegimeDetectorConfig(
            window=60,
            thresholds=thresholds,
        )
        self.regime_detector = RegimeDetector(rd_cfg)
        self._current_regime = None

        # -------------------------------------------------------------------
        # RewardEngine stub (if ai.reward.reward_engine does not exist)
        # -------------------------------------------------------------------
        try:
            from ai.reward.reward_engine import RewardEngine
        except Exception:
            class RewardEngine:
                def __init__(self, *args, **kwargs):
                    pass
        
                def compute_reward(self, info: dict) -> float:
                    """
                    Safe fallback reward engine.
                    Always returns 0.0 unless replaced by real implementation.
                    """
                    return 0.0
        
            print("⚠️  Using RewardEngine stub (ai.reward.reward_engine not found)")


        self.reward_engine = RewardEngine(
            config_path="configs/phase90_reward.yaml",
        )
        self.replay_path = Path("data/replay/phase26_replay.jsonl")
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Autoscaler harness (Phase 26 Ultra)
        # -------------------------------------------------------------
        self.autoscaler = AdaptiveAutoScalingExecutor(
            base_interval_sec=float(os.getenv("PHASE26_BASE_INTERVAL", "5.0")),
            min_interval_sec=float(os.getenv("PHASE26_MIN_INTERVAL", "1.0")),
            max_interval_sec=float(os.getenv("PHASE26_MAX_INTERVAL", "60.0")),
        )

        # -------------------------------------------------------------
        # State
        # -------------------------------------------------------------
        self.running = True
        self._alpha_price_hist: Dict[str, deque[float]] = {
            sym: deque(maxlen=200) for sym in self.symbols
        }

        self._ticks_since_trade = 0
        self._error_ticks = 0

        # Phase 114 – counters
        self.rl_enabled = True
        self.orders_enabled = True

        # Phase 112 – store bot's intended state for reconciliation
        self._bot_state: Dict[str, Any] = {}
        self._bot_state_path = Path("data/runtime/phase26_bot_state.json")
        self._bot_state_path.parent.mkdir(parents=True, exist_ok=True)

        # Phase 26/92 – log basic config
        log.info(
            "Phase26RealtimeUltra: MODE=%s ENV=%s symbols=%s",
            self.mode,
            self.env,
            self.symbols,
        )

        # Phase 111 – log guardian config
        log.info(
            "Phase26RealtimeUltra: guardian_enabled=%s health_monitor_enabled=%s",
            self.guardian_enabled,
            self.health_monitor_enabled,
        )

    # =================================================================
    # Helper: Kill-switch check with heartbeat monitoring
    # =================================================================
    def _check_kill_switch(self) -> bool:
        """Check kill switch state and update heartbeat."""
        try:
            from ai.guardian.kill_switch_monitor import KillSwitchMonitor, KillSwitchConfig
            
            # Initialize monitor if not already done
            if not hasattr(self, "_kill_switch_monitor"):
                config = KillSwitchConfig(
                    flag_path=str(self.kill_flag_path),
                    heartbeat_interval_sec=30.0,
                    heartbeat_timeout_sec=120.0,
                    auto_activate_on_critical_error=True,
                )
                self._kill_switch_monitor = KillSwitchMonitor(config)
            
            # Check kill switch state
            state_info = self._kill_switch_monitor.check()
            
            # Log state changes
            if state_info.get("state_changed"):
                if state_info["active"]:
                    log.error(
                        "🚨 KILL SWITCH ACTIVATED (path: %s)",
                        self.kill_flag_path
                    )
                else:
                    log.warning(
                        "⚠️ KILL SWITCH DEACTIVATED (path: %s)",
                        self.kill_flag_path
                    )
            
            # Check heartbeat
            heartbeat_status = self._kill_switch_monitor.get_heartbeat_status()
            if not heartbeat_status.get("heartbeat_ok"):
                log.warning(
                    "⚠️ Kill switch heartbeat stale: %s",
                    heartbeat_status.get("time_since_heartbeat", "unknown")
                )
            
            # Update internal state
            self._kill_flag_was_active = state_info["active"]
            return state_info["active"]
            
        except Exception as e:
            log.error("Failed to check kill switch: %s", e, exc_info=True)
            # Fallback to simple file check
            exists = self.kill_flag_path.exists()
            if exists and not self._kill_flag_was_active:
                log.error(
                    "⚠️ Kill-switch flag detected at %s; entering monitor-only mode.",
                    self.kill_flag_path,
                )
                self._kill_flag_was_active = True
            elif not exists and self._kill_flag_was_active:
                log.warning(
                    "Kill-switch flag cleared at %s; trading may resume.",
                    self.kill_flag_path,
                )
                self._kill_flag_was_active = False
            return exists

    # =================================================================
    # Alpha features / price history
    # =================================================================
    def _update_alpha_features(self, symbol: str, price: float) -> Dict[str, float]:
        """
        Build the full Phase 102 feature vector that AlphaModel was trained on.

        Training features (from phase102_features.csv):
            open, high, low, close, volume,
            ret_1, ret_5, ret_10,
            vol_10, vol_20,
            ma_10, ma_ratio_10,
            ma_20, ma_ratio_20,
            ma_50, ma_ratio_50,
            rsi_14

        We approximate OHLC/volume from the intrabar price history so that
        runtime features line up by *name* with the training schema.
        """
        hist = self._alpha_price_hist[symbol]
        hist.append(price)

        prices = np.array(hist, dtype=float)
        feats: Dict[str, float] = {}

        # --- OHLC approximations -------------------------------------
        if len(prices) == 0:
            feats["open"] = float(price)
            feats["high"] = float(price)
            feats["low"] = float(price)
        else:
            feats["open"] = float(prices[0])
            feats["high"] = float(prices.max())
            feats["low"] = float(prices.min())
        feats["close"] = float(price)

        # We do not have live volume here; use a neutral constant.
        feats["volume"] = 0.0

        # --- Returns & volatility ------------------------------------
        if len(prices) >= 2:
            rets = np.diff(np.log(prices))

            # ret_1, ret_5, ret_10: cumulative log returns
            feats["ret_1"] = float(rets[-1])
            feats["ret_5"] = float(rets[-5:].sum()) if len(rets) >= 5 else float(rets.sum())
            feats["ret_10"] = float(rets[-10:].sum()) if len(rets) >= 10 else float(rets.sum())

            # vol_10 / vol_20: annualized volatility of log returns
            if len(rets) >= 10:
                feats["vol_10"] = float(np.std(rets[-10:]) * np.sqrt(252.0))
            else:
                feats["vol_10"] = float(np.std(rets) * np.sqrt(252.0))

            if len(rets) >= 20:
                feats["vol_20"] = float(np.std(rets[-20:]) * np.sqrt(252.0))
            else:
                feats["vol_20"] = float(np.std(rets) * np.sqrt(252.0))

            # --- Moving averages --------------------------------------
            window10 = min(len(prices), 10)
            window20 = min(len(prices), 20)
            window50 = min(len(prices), 50)

            ma10 = float(prices[-window10:].mean()) if window10 > 0 else float(price)
            ma20 = float(prices[-window20:].mean()) if window20 > 0 else float(price)
            ma50 = float(prices[-window50:].mean()) if window50 > 0 else float(price)

            feats["ma_10"] = ma10
            feats["ma_ratio_10"] = float(price / ma10) if ma10 != 0 else 1.0
            feats["ma_20"] = ma20
            feats["ma_ratio_20"] = float(price / ma20) if ma20 != 0 else 1.0
            feats["ma_50"] = ma50
            feats["ma_ratio_50"] = float(price / ma50) if ma50 != 0 else 1.0

            # --- RSI(14) on price ------------------------------------
            deltas = np.diff(prices)
            window = min(14, len(deltas))
            if window > 0:
                recent = deltas[-window:]
                gains = np.where(recent > 0, recent, 0.0)
                losses = np.where(recent < 0, -recent, 0.0)
                avg_gain = gains.mean()
                avg_loss = losses.mean()
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / max(avg_loss, 1e-8)
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                feats["rsi_14"] = float(rsi)
            else:
                feats["rsi_14"] = 50.0
        else:
            # Not enough history yet; default neutrals.
            feats["ret_1"] = 0.0
            feats["ret_5"] = 0.0
            feats["ret_10"] = 0.0
            feats["vol_10"] = 0.0
            feats["vol_20"] = 0.0
            feats["ma_10"] = float(price)
            feats["ma_ratio_10"] = 1.0
            feats["ma_20"] = float(price)
            feats["ma_ratio_20"] = 1.0
            feats["ma_50"] = float(price)
            feats["ma_ratio_50"] = 1.0
            feats["rsi_14"] = 50.0

        # Keep old 'price' field for any other consumers in the loop.
        feats["price"] = float(price)

        return feats

    # =================================================================
    # Portfolio Snapshot (RiskEnvelope + Fusion context)
    # =================================================================
    def _portfolio_snapshot(self) -> Dict[str, Any]:
        ext = getattr(self, "_external_portfolio_snapshot", None)
        if isinstance(ext, dict) and ext.get("equity") is not None:
            # Replay/backtest path: caller supplies a deterministic snapshot.
            try:
                eq = float(ext.get("equity") or 0.0)
            except Exception:
                eq = 0.0
            return {
                "equity": eq,
                "positions": ext.get("positions", {}) or {},
                "drawdown": float(ext.get("drawdown") or 0.0),
                "vol": float(ext.get("vol") or 0.0),
            }
        # Option A – use DemoBroker snapshots when running in DEMO mode so
        # risk engines & fusion see the simulated portfolio rather than the
        # real Alpaca account.
        if getattr(self, "demo_mode", False) and getattr(self, "demo_broker", None) is not None:  # noqa: E501
            acct = self.demo_broker.get_account()
            positions_iter = self.demo_broker.get_positions()
        else:
            acct = self.broker.get_account()
            positions_iter = self.broker.get_positions()

        pos_map: Dict[str, Dict[str, float]] = {}
        for p in positions_iter:
            sym = getattr(p, "symbol", None)
            if not sym:
                continue
            pos_map[sym] = {
                "qty": float(getattr(p, "qty", 0.0)),
                "price": float(getattr(p, "current_price", 0.0)),
            }

        # -----------------------------
        # Equity (source of truth)
        # -----------------------------
        equity = 0.0
        
        if self.demo_mode and getattr(self, "demo_broker", None) is not None:
            acct = self.demo_broker.get_account()
            equity = float(getattr(acct, "equity", 0.0) or 0.0)
        else:
            # PAPER/LIVE - AlpacaClient.get_account() returns a dict
            if isinstance(acct, dict):
                equity = float(acct.get("equity", 0.0) or 0.0)
            else:
                equity = float(getattr(acct, "equity", 0.0) or 0.0)
        
        # Final sanity
        if not np.isfinite(equity) or equity <= 0:
            equity = 0.0


        # --------------------------------------------------
        # HARD equity sanity clamp (CRITICAL FIX)
        # --------------------------------------------------
        MAX_REASONABLE_EQUITY = 1e7   # $10M hard ceiling (DEMO-safe)
        
        if not np.isfinite(equity) or equity <= 0 or equity > MAX_REASONABLE_EQUITY:
            if getattr(self, "_last_good_equity", 0.0) >= 1000:
                log.warning(
                    "Equity invalid (%.3e); reverting to last_good_equity=%.2f",
                    equity,
                    self._last_good_equity,
                )
                equity = float(self._last_good_equity)
            else:
                self._log_equity_invalid_throttled(equity)
                equity = 0.0
        else:
            self._last_good_equity = equity



        try:
            dd = float(self.pbrain.get_drawdown() or 0.0)
        except Exception:
            dd = 0.0

        try:
            vol = float(self.pbrain.get_realized_vol() or 0.0)
        except Exception:
            vol = 0.0

        return {
            "equity": equity,
            "positions": pos_map,
            "drawdown": dd,
            "volatility": vol,
            "aggression_factor": float(self._aggression_factor),
            "mode": self.mode,
        }

    # =================================================================
    # Aggression / Regime Update (Phase 69E)
    # =================================================================
    def _update_aggression(self) -> None:
        if getattr(self, "demo_mode", False) and getattr(self, "demo_broker", None) is not None:  # noqa: E501
            acct = self.demo_broker.get_account()
            equity = float(getattr(acct, "equity", 0.0))
        else:
            acct = self.broker.get_account()
            # AlpacaClient.get_account() returns a dict
            if isinstance(acct, dict):
                equity = float(acct.get("equity", 0.0) or 0.0)
            else:
                equity = float(getattr(acct, "equity", 0.0) or 0.0)
        self._equity_curve.append(equity)

        if len(self._equity_curve) < 5:
            self._aggression_factor = 1.0
            return

        peak = max(self._equity_curve)
        dd = (peak - equity) / max(peak, 1.0)

        factor, meta = self.equity_agent.step(list(self._equity_curve), dd)
        self._aggression_factor = factor

        log.info(
            "🎯 AggFactor=%.2f | Regime=%s | DD=%.2f%% | Slope=%.6f",
            factor,
            meta.get("regime"),
            dd * 100.0,
            meta.get("slope"),
        )

    # =================================================================
    # Phase 92.4 – Stability parameter computation
    # =================================================================
    def _compute_stability_params(self) -> Dict[str, Any]:
        """
        Derive volatility/drawdown-aware parameters for this tick:

            - entry_threshold: fused score threshold to consider trading
            - max_equity_risk_pct: per-symbol equity allocation cap
            - symbol_cooldown_sec: effective per-symbol cooldown
        """
        snapshot = self._portfolio_snapshot()
        vol = float(snapshot.get("volatility", 0.0))
        dd = float(snapshot.get("drawdown", 0.0))

        base_thr = float(os.getenv("PHASE26_BASE_ENTRY_THR", "0.08"))
        base_max_risk = float(os.getenv("PHASE26_MAX_EQUITY_RISK_PCT", "0.03"))
        base_cooldown = float(os.getenv("PHASE26_SYMBOL_COOLDOWN_SEC", "60.0"))

        if vol < 0.01:
            vol = 0.01

        entry_threshold = base_thr * (1.0 + vol * 0.5 + dd * 2.0)
        max_equity_risk_pct = base_max_risk / (1.0 + vol + dd * 2.0)
        symbol_cooldown_sec = base_cooldown * (1.0 + vol * 1.5 + dd * 3.0)

        entry_threshold = float(np.clip(entry_threshold, 0.02, 0.25))
        max_equity_risk_pct = float(np.clip(max_equity_risk_pct, 0.01, 0.05))
        symbol_cooldown_sec = float(np.clip(symbol_cooldown_sec, 10.0, 300.0))

        return {
            "entry_threshold": entry_threshold,
            "max_equity_risk_pct": max_equity_risk_pct,
            "symbol_cooldown_sec": symbol_cooldown_sec,
        }

    # =================================================================
    # Helper: autoscaler state
    # =================================================================
    def _autoscale_state(self) -> Dict[str, Any]:
        try:
            return {"volatility": float(self.pbrain.get_realized_vol())}
        except Exception:
            return {"volatility": 1.0}



    # =================================================================
    # Phase E: market-open awareness + operator-safe idling
    # =================================================================
    def _ny_now(self) -> datetime:
        return datetime.now(ZoneInfo("America/New_York"))

    def _is_market_open_now(self) -> bool:
        """Best-effort market-open check.

        Order of truth:
          1) Alpaca clock (if broker exposes get_clock)
          2) (Optional) pandas_market_calendars NYSE schedule
          3) Fallback: weekday 9:30-16:00 ET
        """
        # DEMO is always "open" (offline simulation)
        if self.mode.upper() == "DEMO":
            return True

        # 1) Broker clock
        try:
            clk = getattr(self.broker, "get_clock", None)
            if callable(clk):
                c = clk()
                is_open = getattr(c, "is_open", None)
                if is_open is None and isinstance(c, dict):
                    is_open = c.get("is_open")
                if isinstance(is_open, str):
                    is_open = is_open.strip().lower() == "true"
                if is_open is not None:
                    return bool(is_open)
        except Exception:
            pass

        # 2) Exchange calendar (optional dependency)
        try:
            import pandas_market_calendars as mcal  # type: ignore

            cal = mcal.get_calendar("NYSE")
            now = self._ny_now()
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return False
            m_open = sched.iloc[0]["market_open"]
            m_close = sched.iloc[0]["market_close"]
            # pandas_market_calendars returns tz-aware timestamps; compare in ET
            now_ts = now
            try:
                o = m_open.tz_convert("America/New_York").to_pydatetime()
                c = m_close.tz_convert("America/New_York").to_pydatetime()
            except Exception:
                o = m_open.to_pydatetime()
                c = m_close.to_pydatetime()
            return o <= now_ts <= c
        except Exception:
            pass

        # 3) Simple fallback
        now = self._ny_now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return (t >= dtime(9, 30)) and (t <= dtime(16, 0))

    def _idle(self, *, reason: str, msg: str, sleep_sec: float = 60.0, log_every_sec: float = 300.0) -> None:
        """Idle with throttled logs (prevents overnight log spam)."""
        now = time.time()
        last = float(getattr(self, f"_last_idle_{reason}", 0.0) or 0.0)
        if (now - last) >= float(log_every_sec):
            log.info(msg)
            setattr(self, f"_last_idle_{reason}", now)
        time.sleep(float(sleep_sec))

    def _operator_safe_equity(self) -> float:
        """Return equity, but never fabricate in PAPER/LIVE."""
        try:
            if self.demo_mode and self.demo_broker is not None:
                return float(self.demo_broker.equity())
        except Exception:
            pass
        try:
            acct = self.broker.get_account()
            eq = getattr(acct, "equity", None)
            if eq is None and isinstance(acct, dict):
                eq = acct.get("equity")
            return float(eq or 0.0)
        except Exception:
            return 0.0

    def _log_equity_invalid_throttled(self, equity: float) -> None:
        """Avoid log spam when broker equity is temporarily unavailable."""
        now = time.time()
        last = float(getattr(self, "_equity_invalid_last_log_ts", 0.0) or 0.0)
        if now - last >= 30.0:
            log.error(
                "Equity invalid and no safe fallback available (%.3e); forcing skip.",
                equity,
            )
            self._equity_invalid_last_log_ts = now

    # =================================================================
    # Helper: price access (real vs synthetic feed)
    # =================================================================
    def _get_last_price(self, symbol: str) -> float | None:
        """Return last price.

        # Replay/backtest: allow injected prices.
        replay_prices = getattr(self, "_replay_last_prices", None)
        if isinstance(replay_prices, dict) and symbol in replay_prices:
            try:
                return float(replay_prices[symbol])
            except Exception:
                return None


        Phase E safety:
        - Synthetic feed is DEMO-only (offline simulation).
        - PAPER/LIVE must use broker prices; on failure we skip the symbol.
        """
        if (
            self.mode.upper() == "DEMO"
            and getattr(self, "synthetic_prices_enabled", False)
            and getattr(self, "synthetic_feed", None) is not None
        ):
            try:
                return float(self.synthetic_feed.next_price(symbol))  # type: ignore[arg-type]  # noqa: E501
            except Exception:
                log.exception(
                    "SyntheticPriceFeed failure for %s; falling back to broker.",
                    symbol,
                )

        try:
            return float(self.broker.get_last_price(symbol))
        except Exception:
            log.exception("Broker price fetch failed for %s; skipping symbol.", symbol)
            return None

    def _on_replay_bars(self, bars: dict) -> None:
        """Drive one tick using externally supplied OHLCV bars.

        bars: {symbol: {"close": float, ...}} OR {symbol: float}
        """
        self._replay_mode = True
        self._disable_live_execution = True
        self._replay_fills = []
        prices = {}
        for sym, v in (bars or {}).items():
            try:
                if isinstance(v, dict):
                    prices[sym] = float(v.get("close") or v.get("price") or 0.0)
                else:
                    prices[sym] = float(v)
            except Exception:
                continue
        self._replay_last_prices = prices
        self._tick({})

    # =================================================================
    # Helper: flatten all positions via SmartOrderRouter (Phase 111)
    # =================================================================
    def _flatten_positions_via_router(self, snapshot: Dict[str, Any]) -> None:
        positions = snapshot.get("positions", {})
        if not positions:
            log.info("LiveCapitalGuardian: no positions to flatten.")
            return

        equity = float(snapshot.get("equity", 0.0))
        log.warning(
            "LiveCapitalGuardian: flatten_all_positions start (equity=%.2f, %d positions).",  # noqa: E501
            equity,
            len(positions),
        )

        for sym, pos in positions.items():
            try:
                qty = float(pos.get("qty", 0.0))
                if qty == 0.0:
                    continue
                side = "SELL" if qty > 0 else "BUY"
                self.router.route_order(
                    symbol=sym,
                    side=side,
                    qty=abs(qty),
                    order_type="MARKET",
                    tag="live_guardian_flatten",
                    risk_ctx=snapshot,
                )
                log.warning(
                    "LiveCapitalGuardian: flattened %s %s via SmartOrderRouter.",
                    qty,
                    sym,
                )
            except Exception:
                log.exception(
                    "LiveCapitalGuardian: failed to flatten position in %s", sym
                )

    # =================================================================
    # Core fused score computation for a symbol
    # =================================================================
    def _compute_fused_score_for_symbol(self, symbol: str, price: float) -> Dict[str, Any]:  # noqa: E501
        """
        Phase 26 + Phase 101 compatible fused-score computation.
    
        Integrates:
          • AlphaModel V2 (RandomForestRegressor via FeatureFactory)
          • Momentum signal
          • Aggression factor
          • Optional PPO / AlphaZoo signals
          • Feature sensitivity scaling
          • Graceful degradation if ML fails
        """
    
        # ============================================================
        # 1) ML FEATURE GENERATION (Phase 101)
        # ============================================================
        ml_pred = 0.0
        ml_score = 0.0
    
        if self._alpha_enabled and self.alpha_model is not None:
            # Modern AlphaModel expects a dict of feature_cols → values.
            try:
                # Phase 101 feature handler: ensure features exist in price histogram
                feats = self._update_alpha_features(symbol, price)
    
                # New AlphaModel interface:
                #   pred, ml_score = predict_one(symbol, feats)
                out = self.alpha_model.predict_one(symbol, feats)
    
                # AlphaModel supports legacy OR new API:
                if isinstance(out, tuple):
                    ml_pred, ml_score = out
                else:
                    # Legacy fallback
                    ml_pred = float(out)
                    ml_score = float(np.tanh(ml_pred * 3.0))
    
            except Exception as e:
                log.warning("AlphaModel prediction failed for %s: %s", symbol, e)
                # Graceful degradation: do NOT force ml_score to zero.
                ml_score = 0.0
    
        # Sensitivity scaling (Phase 26 Options B/C)
        if getattr(self, "_ml_sensitivity", 1.0) != 1.0:
            ml_score *= float(self._ml_sensitivity)
    
        # Clamp ML score into [-1, 1]
        ml_score = float(np.clip(ml_score, -1.0, 1.0))
    
    
        # ============================================================
        # 2) MOMENTUM SIGNAL
        # ============================================================
        mom_score = 0.0
        hist = self._alpha_price_hist[symbol]
    
        if len(hist) >= 5:
            try:
                prices = np.array(hist, dtype=float)
                returns = np.diff(np.log(prices))
                # Use last 3 bars momentum
                raw_mom = returns[-3:].sum()
                mom_score = float(np.tanh(raw_mom * 50.0))
            except Exception as e:
                log.warning("Momentum calc failed for %s: %s", symbol, e)
                mom_score = 0.0
    
        if getattr(self, "_mom_scale", 1.0) != 1.0:
            mom_score *= float(self._mom_scale)
    
        mom_score = float(np.clip(mom_score, -1.0, 1.0))
    
    
        # ============================================================
        # 3) AGGRESSION SCORE (Phase 69E)
        # ============================================================
        agg_score = float(self._aggression_factor - 1.0)
    
    
        # ============================================================
        # 4) FUSION CONTEXT (Phase 88/89/92/123)
        # ============================================================
        fusion_ctx = self._portfolio_snapshot()
        try:
            fusion_ctx["volatility"] = float(self.pbrain.get_realized_vol() or 1.0)
        except Exception:
            fusion_ctx["volatility"] = 1.0
    
    
        # ============================================================
        # 5) FUSE SIGNALS (FusionBrain or FusionEngine)
        # ============================================================
        if hasattr(self, "fusion_engine") and self.fusion_engine is not None:
            fused_out = self.fusion_engine.fuse(
                symbol=symbol,
                price=price,
                ml_pred=ml_pred,
                ml_score=ml_score,
                mom_score=mom_score,
                agg_score=agg_score,
                ctx=fusion_ctx,
            )
        else:
            fused_out = self.fusion_brain.fuse(
                symbol=symbol,
                price=price,
                ml_pred=ml_pred,
                ml_score=ml_score,
                mom_score=mom_score,
                agg_score=agg_score,
                ctx=fusion_ctx,
            )
    
    
        # ============================================================
        # 6) EXTRACT SCORES & SAFETY DEFAULTS
        # ============================================================
        fused_score = float(
            fused_out.get("score", fused_out.get("fused_score", 0.0))
        )
        ensemble_score = float(fused_out.get("ensemble_score", fused_score))
        fused_weights = fused_out.get("weights", {})
    
        # Clamp fused score for extra safety
        fused_score = float(np.clip(fused_score, -1.0, 1.0))
        ensemble_score = float(np.clip(ensemble_score, -1.0, 1.0))
    
        # ============================================================
        # 7) RETURN STRUCTURE (Phase 26 expects this format)
        # ============================================================
        return {
            "symbol": symbol,
            "price": price,
            "fused_score": fused_score,
            "ensemble_score": ensemble_score,
            "ml_score": float(ml_score),
            "ml_pred": float(ml_pred),
            "mom_score": float(mom_score),
            "agg_score": float(agg_score),
            "ppo_val": float(fused_out.get("ppo_val", 0.0)),
            "alphazoo_val": float(fused_out.get("alphazoo_val", 0.0)),
            "weights": fused_weights,
        }


    def _maybe_hotswap_models(self, ctx: Dict[str, Any]) -> None:
        """
        Phase B: Safely reload AlphaModel (and optional policies) if files changed.
        - Never stops the loop
        - Never swaps in a model unless it loads AND validates
        """
        if not getattr(self, "hotswap_enabled", False):
            return
    
        now = time.time()
        if (now - getattr(self, "hotswap_last_check_ts", 0.0)) < getattr(self, "hotswap_check_sec", 10.0):
            return
        self.hotswap_last_check_ts = now
    
        # -----------------------------
        # AlphaModel hot-swap
        # -----------------------------
        try:
            alpha_path = getattr(getattr(self, "alpha_model_cfg", None), "model_path", None)
            if alpha_path:
                p = Path(alpha_path)
                mtime = p.stat().st_mtime if p.exists() else 0.0
                if mtime > 0 and mtime != getattr(self, "_alpha_model_mtime", 0.0):
                    log.warning("🔁 HotSwap: AlphaModel file changed → reloading (%s)", alpha_path)
    
                    # Load candidate model FIRST (no mutation yet)
                    candidate = AlphaModel(self.alpha_model_cfg)
    
                    # Validate candidate (must produce a numeric score on a real feature vector)
                    # Use the first symbol from live set
                    sym = self.symbols[0] if getattr(self, "symbols", None) else "AAPL"
                    px = float(self._get_last_price(sym) or 100.0)
                    feats = self._update_alpha_features(sym, px)
                    out = candidate.predict_one(sym, feats)
    
                    # Validate output type
                    if isinstance(out, tuple):
                        _, score = out
                        score = float(score)
                    else:
                        score = float(out)
    
                    if not np.isfinite(score):
                        raise ValueError(f"candidate AlphaModel produced non-finite score: {score}")
    
                    # Atomic swap
                    with self._hotswap_lock:
                        self.alpha_model = candidate
                        self._alpha_enabled = True
                        self._alpha_model_mtime = mtime
    
                    log.warning("✅ HotSwap: AlphaModel swapped in successfully (score=%.4f)", score)
    
        except Exception as e:
            log.error("HotSwap AlphaModel failed; keeping existing model. err=%s", e, exc_info=True)
    
        # -----------------------------
        # Optional policy hot-swap (best-effort)
        # -----------------------------
        if not getattr(self, "hotswap_policy_paths", None):
            return
    
        for path in list(self.hotswap_policy_paths):
            try:
                pp = Path(path)
                mtime = pp.stat().st_mtime if pp.exists() else 0.0
                last = float(self._policy_mtimes.get(path, 0.0) or 0.0)
                if mtime > 0 and mtime != last:
                    log.warning("🔁 HotSwap: policy file changed → %s", path)
    
                    # Best-effort: if supervisor has a reload hook, call it; otherwise no-op.
                    # We will not break older phases.
                    with self._hotswap_lock:
                        if hasattr(self, "policy_supervisor") and hasattr(self.policy_supervisor, "reload"):
                            self.policy_supervisor.reload()
                            log.warning("✅ HotSwap: policy_supervisor.reload() executed.")
                        else:
                            log.warning("HotSwap: No policy reload hook available (safe no-op).")
    
                    self._policy_mtimes[path] = mtime
    
            except Exception as e:
                log.error("HotSwap policy failed (%s): %s", path, e, exc_info=True)

    # =================================================================
    # Tick function
    # =================================================================
    def _tick(self, ctx: Dict[str, Any]) -> None:
        start_ts = time.time()
        # Replay/backtest mode: bypass market-hours and operator equity gating.
        if getattr(self, "_replay_mode", False):
            pass
        # Phase E operator-safety: only trade during market hours (PAPER/LIVE)
        elif self.mode.upper() != "DEMO":
            if not self._is_market_open_now():
                self._idle(reason="market_closed", msg="Market closed — idling.", sleep_sec=60.0)
                return

            eq = self._operator_safe_equity()
            if eq <= 0.0:
                self._idle(
                    reason="equity_unavailable",
                    msg="Broker equity unavailable (<=0). Idling and retrying.",
                    sleep_sec=60.0,
                )
                return

        # Phase B: hot-swap models/policies safely (no downtime)
        self._maybe_hotswap_models(ctx)

        log.info("⏱ Tick #%s (%.3fs)", ctx.get("tick_idx"), ctx.get("interval_sec"))

        # Phase 114 – track per-tick health flags
        tick_error = False
        had_trade = False

        # -------------------------------------------------------------------
        # Kill-switch handling: monitor-only mode while flag exists
        # -------------------------------------------------------------------
        flag_exists = self._check_kill_switch()
        # ---------------------------------------------------------
        # Phase 26 DEMO MODE – disable legacy file-based kill-switch
        # ---------------------------------------------------------
        if self.mode.upper() == "DEMO":
            log.info(
                "DEMO mode: ignoring legacy kill-switch flag (monitor-only kill disabled)."  # noqa: E501
            )
            flag_exists = False

        if flag_exists:
            log.warning(
                "Kill-switch flag present: running in monitor-only mode (no trades)."
            )

        try:
            # -------------------------------------------------------------
            # Update aggression factor (Phase 69E)
            # -------------------------------------------------------------
            self._update_aggression()

            # -------------------------------------------------------------
            # Phase 111 – LiveCapitalGuardian check
            # -------------------------------------------------------------
            guardian_decision = None
            if self.guardian_enabled and not self.demo_mode:
                try:
                    snapshot_for_guardian = self._portfolio_snapshot()
                    # Real implementation uses .check(), stub uses .evaluate()
                    if hasattr(self.live_cap_guardian, 'check'):
                        equity = snapshot_for_guardian.get('equity', 0.0)
                        positions = snapshot_for_guardian.get('positions', [])
                        guardian_decision = self.live_cap_guardian.check(
                            equity=equity,
                            positions=positions
                        )
                    else:
                        guardian_decision = self.live_cap_guardian.evaluate(
                            portfolio_snapshot=snapshot_for_guardian
                        )
                except Exception as e:
                    log.error(
                        "LiveCapitalGuardian evaluation failed: %s",
                        e,
                        exc_info=True,
                    )

                if (
                    self.guardian_enabled
                    and guardian_decision is not None
                    and guardian_decision.kill_switch_active
                ):
                    msg = (
                        f"🚨 LiveCapitalGuardian KILL-SWITCH ACTIVE\n"
                        f"Reason: {guardian_decision.reason}\n"
                        f"Metrics: {guardian_decision.metrics}"
                    )
                    log.error(msg)
                    try:
                        notify(
                            msg,
                            kind="guardian",
                            meta={
                                "phase": 26,
                                "snapshot": snapshot_for_guardian,
                            },
                        )
                    except Exception:
                        log.exception("Failed to send LiveCapitalGuardian alert.")

                    self.running = False
                    return

                if (
                    self.guardian_enabled
                    and guardian_decision is not None
                    and guardian_decision.should_flatten
                ):
                    try:
                        snapshot_for_guardian = (
                            snapshot_for_guardian or self._portfolio_snapshot()
                        )
                        self._flatten_positions_via_router(snapshot_for_guardian)
                    except Exception:
                        log.exception(
                            "LiveCapitalGuardian flatten_all_positions failed.",
                            exc_info=True,
                        )

                if (
                    self.guardian_enabled
                    and guardian_decision is not None
                    and guardian_decision.disable_new_orders
                ):
                    self.orders_enabled = False

                    log.warning(
                        "LiveCapitalGuardian: orders disabled by guardian; monitor-only mode."  # noqa: E501
                    )
                    return

            # If guardian is OK but RL or orders are disabled from previous trip,
            # remain in monitoring mode (no trades) for now.
            if not self.rl_enabled or not self.orders_enabled:
                log.warning(
                    "Phase 26: RL or orders disabled by LiveCapitalGuardian; monitoring-only mode."  # noqa: E501
                )
                return

            # -------------------------------------------------------------
            # Phase 26 Stability logic – compute thresholds for this tick
            # -------------------------------------------------------------
            stability = self._compute_stability_params()
            entry_threshold = stability["entry_threshold"]
            max_equity_risk_pct = stability["max_equity_risk_pct"]
            symbol_cooldown_sec = stability["symbol_cooldown_sec"]

            # Phase 26 DEMO option: disable entry threshold completely
            if self.mode.upper() == "DEMO":
                log.info(
                    "DEMO mode: overriding entry_threshold=%.4f → 0.0 (no clamp on signal).",  # noqa: E501
                    entry_threshold,
                )
                entry_threshold = 0.0

            # -------------------------------------------------------------
            # Gather fused scores for each tradable symbol
            # -------------------------------------------------------------
            candidates = []
            demo_price_map: Dict[str, float] = {}
            for sym in self.symbols:
                price = self._get_last_price(sym)
                if price is None:
                    continue
                price_f = float(price)
                # Phase 123 – feed latest price into Meta-Stability engine
                try:
                    if getattr(self, "meta_engine", None) is not None:
                        self.meta_engine.update(price_f)
                except Exception:
                    log.exception("Phase 123: meta_engine.update failed.")
                fused_info = self._compute_fused_score_for_symbol(sym, price_f)
                
                # Phase 2: Trade Quality Filter - filter signals before adding to candidates
                if hasattr(self, 'trade_quality_filter') and self.trade_quality_filter:
                    fused_score = fused_info.get("fused_score", 0.0)
                    signal_type = "ml" if abs(fused_info.get("ml_score", 0.0)) > abs(fused_info.get("mom_score", 0.0)) else "momentum"
                    
                    # Get market data for filtering
                    try:
                        from ai.market.enhanced_data_provider import EnhancedMarketDataProvider
                        provider = EnhancedMarketDataProvider()
                        quote = provider.get_quote(sym)
                        volume = None
                        spread_bps = None
                        if quote:
                            # Estimate volume from quote (if available)
                            # Spread calculation
                            if quote.get("bid") and quote.get("ask") and quote.get("price"):
                                spread = (quote["ask"] - quote["bid"]) / quote["price"] * 10000  # Convert to bps
                                spread_bps = spread
                    except Exception:
                        quote = None
                        volume = None
                        spread_bps = None
                    
                    # Filter signal
                    allowed, reason = self.trade_quality_filter.filter_signal(
                        symbol=sym,
                        signal_type=signal_type,
                        signal_strength=fused_score,
                        price=price_f,
                        volume=volume,
                        bid_ask_spread=spread_bps,
                        volatility=None,  # Could calculate from historical data
                    )
                    
                    if not allowed:
                        log.debug(
                            "Trade quality filter blocked %s: %s (fused=%.4f)",
                            sym, reason, fused_score
                        )
                        continue  # Skip this symbol
                
                log.info(
                    "Signals %s | price=%.2f | fused=%.4f | ml=%.4f | mom=%.4f | agg=%.4f",
                    sym,
                    price_f,
                    fused_info.get("fused_score", 0.0),
                    fused_info.get("ml_score", 0.0),
                    fused_info.get("mom_score", 0.0),
                    fused_info.get("agg_score", 0.0),
                )
                candidates.append(fused_info)
                if self.demo_broker is not None and self.demo_mode:
                    demo_price_map[sym] = price_f

            # Update DemoBroker marks so PnL reflects latest prices
            if self.demo_broker is not None and demo_price_map:
                self.demo_broker.update_mark_prices(demo_price_map)

            if not candidates:
                log.warning("No prices available for any symbol; skipping tick.")
                return

            # -------------------------------------------------------------
            # Phase 123 – Meta-Stability clamp applied to ALL candidates
            # (so allocator sees the risk-adjusted scores)
            # -------------------------------------------------------------
            clamp_factor = 1.0
            try:
                if getattr(self, "meta_engine", None) is not None:
                    meta_decision = self.meta_engine.evaluate()
                    clamp_factor = float(getattr(meta_decision, "clamp_factor", 1.0) or 1.0)
                    if clamp_factor < 1.0:
                        log.info(
                            "MetaStability clamp: all scores scaled by %.3f (reason=%s, metrics=%s)",
                            clamp_factor,
                            getattr(meta_decision, "reason", None),
                            getattr(meta_decision, "metrics", None),
                        )
            except Exception:
                log.exception("MetaStabilityEngine evaluation failed; skipping clamp.")
            
            for c in candidates:
                # Scale ensemble_score if present, else scale fused_score
                if "ensemble_score" in c and c["ensemble_score"] is not None:
                    c["ensemble_score"] = float(c["ensemble_score"]) * clamp_factor
                c["fused_score"] = float(c.get("fused_score", 0.0)) * clamp_factor
            
            
            # -------------------------------------------------------------
            # Snapshot (equity/positions/volatility) for allocators
            # -------------------------------------------------------------
            snapshot_before = self._portfolio_snapshot()
            positions = snapshot_before.get("positions", {})
            equity = float(snapshot_before.get("equity", 0.0) or 0.0)
            
            # Last-known-good equity fallback (protects against transient broker glitches)
            if not hasattr(self, "_last_good_equity"):
                self._last_good_equity = 0.0
            
            if equity > 0:
                if np.isfinite(equity) and 0 < equity < 1e9:
                    self._last_good_equity = equity

            elif self._last_good_equity >= 1000:
                log.warning("Equity read as 0; using last-known-good equity=%.2f", self._last_good_equity)
                equity = float(self._last_good_equity)
            
            
            # If equity is still not usable, skip safely
            if equity <= 0.0:
                log.warning("Equity <= 0; skipping tick (allocator/risk protection).")
                return
            
            # -------------------------------------------------------------
            # Phase 1: Exit Manager - Check for exits on existing positions
            # -------------------------------------------------------------
            exit_orders = []
            if hasattr(self, 'exit_manager') and self.exit_manager:
                try:
                    # Get current prices for all positions
                    for symbol, pos_data in positions.items():
                        qty = float(pos_data.get("qty", 0.0))
                        if abs(qty) < 1e-6:
                            continue  # Skip flat positions
                        
                        current_price = float(pos_data.get("price", 0.0))
                        if current_price <= 0:
                            # Try to get price
                            current_price = self._get_last_price(symbol)
                            if current_price is None:
                                continue
                        
                        # Calculate ATR if needed for trailing stops
                        atr_value = None
                        if self.exit_manager.config.trailing_stop_enabled and \
                           self.exit_manager.config.trailing_stop_method == "atr":
                            # Try to get historical data for ATR
                            try:
                                from ai.market.enhanced_data_provider import EnhancedMarketDataProvider
                                provider = EnhancedMarketDataProvider()
                                hist_data = provider.get_historical_data(symbol, period="1mo", interval="1d")
                                if hist_data and len(hist_data) >= self.exit_manager.config.atr_period + 1:
                                    df = pd.DataFrame(hist_data)
                                    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                                    atr_value = self.exit_manager.calculate_atr(symbol, df)
                            except Exception as e:
                                log.debug("Failed to calculate ATR for %s: %s", symbol, e)
                        
                        # Check for exit signals
                        exit_order = self.exit_manager.update_position(
                            symbol=symbol,
                            current_price=current_price,
                            current_qty=qty,
                            atr=atr_value,
                            signal_reversed=False,  # TODO: detect signal reversal
                            regime_changed=False,  # TODO: detect regime change
                            volatility_spike=False,  # TODO: detect volatility spike
                        )
                        
                        if exit_order:
                            exit_orders.append(exit_order)
                            log.info(
                                "🚪 Exit signal: %s %s %.2f @ %.2f (reason: %s)",
                                exit_order["side"],
                                symbol,
                                exit_order["qty"],
                                current_price,
                                exit_order.get("reason", "unknown")
                            )
                    
                    # Execute exit orders
                    for exit_order in exit_orders:
                        symbol = exit_order["symbol"]
                        side = exit_order["side"]
                        qty = exit_order["qty"]
                        reason = exit_order.get("reason", "exit_signal")
                        
                        # Get current price
                        price = self._get_last_price(symbol)
                        if price is None:
                            log.warning("Cannot execute exit for %s: no price", symbol)
                            continue
                        
                        # Route exit order
                        try:
                            if self.demo_mode and getattr(self, "demo_broker", None) is not None:
                                res = self.demo_broker.place_order(
                                    symbol=symbol,
                                    side=side,
                                    qty=qty,
                                    price=price,
                                    tag=f"exit_{reason}",
                                )
                            else:
                                risk_ctx = self._portfolio_snapshot()
                                res = self.router.route_order(
                                    symbol=symbol,
                                    side=side,
                                    qty=qty,
                                    order_type="MARKET",
                                    tag=f"exit_{reason}",
                                    risk_ctx=risk_ctx,
                                )
                            log.info("✅ Exit order executed: %s %s %.2f @ %.2f → %s", 
                                   side, symbol, qty, price, res)
                            had_trade = True
                        except Exception as e:
                            log.error("Failed to execute exit order for %s: %s", symbol, e, exc_info=True)
                
                except Exception as e:
                    log.error("Exit Manager check failed: %s", e, exc_info=True)
            
            # Register new positions (detect when position changes from 0 to non-zero)
            if hasattr(self, 'exit_manager') and self.exit_manager:
                try:
                    if not hasattr(self, '_last_positions'):
                        self._last_positions = {}
                    
                    for symbol, pos_data in positions.items():
                        qty = float(pos_data.get("qty", 0.0))
                        current_price = float(pos_data.get("price", 0.0))
                        if current_price <= 0:
                            current_price = self._get_last_price(symbol) or 0.0
                        
                        last_qty = self._last_positions.get(symbol, 0.0)
                        
                        # Detect new position (was flat, now has position)
                        if abs(last_qty) < 1e-6 and abs(qty) >= 1e-6:
                            # New position opened
                            entry_price = current_price  # Approximate entry price
                            self.exit_manager.register_position(
                                symbol=symbol,
                                entry_price=entry_price,
                                qty=qty,
                                entry_time=datetime.now(),
                                initial_stop_loss=None,  # Could calculate from ATR
                            )
                            log.info("📝 Registered new position for exit management: %s %.2f @ %.2f", 
                                   symbol, qty, entry_price)
                        
                        # Update last known position
                        self._last_positions[symbol] = qty
                
                except Exception as e:
                    log.debug("Failed to register positions: %s", e)
            
            
            # -------------------------------------------------------------
            # Stronger entry gate: if score below entry_threshold, ignore it (non-DEMO)
            # (Allocator also has its own floors, but this keeps Phase26 behavior consistent)
            # -------------------------------------------------------------
            if not self.demo_mode:
                gated = []
                for c in candidates:
                    s = float(c.get("ensemble_score", c.get("fused_score", 0.0)) or 0.0)
                    if abs(s) >= float(entry_threshold):
                        gated.append(c)
                candidates = gated
            
                if not candidates:
                    log.info(
                        "All candidate scores below entry_threshold %.4f; skipping tick.",
                        float(entry_threshold),
                    )
                    return
            
            
            # -------------------------------------------------------------
            # Phase 28: Multi-symbol allocation (portfolio-aware)
            # -------------------------------------------------------------
            # Use portfolio volatility if present; fallback to 1.0
            portfolio_vol = float(snapshot_before.get("volatility", 1.0) or 1.0)
            volatility_map = {c["symbol"]: portfolio_vol for c in candidates}
            
            decisions = []
            try:
                if getattr(self, "multi_alloc", None) is not None:
                    decisions = self.multi_alloc.allocate(
                        equity=equity,
                        positions=positions,
                        candidates=candidates,
                        volatility_map=volatility_map,
                        mode=self.mode,
                        ctx={"tick_idx": ctx.get("tick_idx")},
                    )
                else:
                    log.warning("multi_alloc not initialized; falling back to single-symbol selection.")
            except Exception:
                log.exception("MultiSymbolAllocator.allocate failed; falling back to single-symbol selection.")
            
            
            # ============================================================
            # Phase A — Multi-symbol execution
            # ============================================================
            
            if decisions and PHASE28_EXECUTE_ALL:
                log.info(
                    "🚀 Phase A enabled: executing %d Phase 28 allocation decisions",
                    len(decisions),
                )
            
                # Execute ALL allocator decisions (ranked)
                for exec_idx, d in enumerate(decisions, start=1):
                    symbol = d.symbol
                    alloc_decision = d.alloc
                    fused_score_raw = float(d.fused_score)
                    decision_score = float(d.ensemble_score)
            
                    # Resolve price from candidate list
                    price = next(
                        (float(c["price"]) for c in candidates if c["symbol"] == symbol),
                        None,
                    )
                    if price is None or price <= 0:
                        log.warning(
                            "[%d] Invalid price for %s; skipping execution.",
                            exec_idx,
                            symbol,
                        )
                        continue
            
                    # --- current position ---
                    cur_pos = positions.get(symbol, {"qty": 0.0, "price": price})
                    cur_qty = float(cur_pos.get("qty", 0.0))
                    desired_qty = float(alloc_decision.target_qty)
                    diff_qty = desired_qty - cur_qty
            
                    log.info(
                        "[%d] %s | fused=%.4f decision=%.4f cur=%.4f tgt=%.4f diff=%.4f",
                        exec_idx,
                        symbol,
                        fused_score_raw,
                        decision_score,
                        cur_qty,
                        desired_qty,
                        diff_qty,
                    )
            
                    # --- noise guard ---
                    if abs(diff_qty) < 0.01 and not self.demo_mode:
                        log.info(
                            "[%d] %s diff %.4f too small; skipping.",
                            exec_idx,
                            symbol,
                            diff_qty,
                        )
                        continue
            
                    side = alloc_decision.side or ("BUY" if diff_qty > 0 else "SELL")
                    order_qty = abs(diff_qty)
            
                    # --- cooldown guard (existing logic preserved) ---
                    cooldown_key = f"{symbol}_last_trade_ts"
                    last_trade_ts = getattr(self, cooldown_key, 0.0)
                    now_ts = time.time()
                    if (
                        not self.demo_mode
                        and (now_ts - last_trade_ts) < symbol_cooldown_sec
                    ):
                        log.info(
                            "[%d] %s under cooldown (%.1fs left); skipping.",
                            exec_idx,
                            symbol,
                            symbol_cooldown_sec - (now_ts - last_trade_ts),
                        )
                        continue
            
                    setattr(self, cooldown_key, now_ts)
            
                    # --- risk envelope ---
                    risk_ctx = self._portfolio_snapshot()
                    if self.guardrails_enabled:
                        if not self.risk_envelope.allow_order(
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            price=price,
                            ctx=risk_ctx,
                        ):
                            log.warning(
                                "[%d] RiskEnvelope blocked %s %s %.4f @ %.2f",
                                exec_idx,
                                side,
                                symbol,
                                order_qty,
                                price,
                            )
                            continue
            
                    # --- route order ---
                    had_trade = True
                    if self.demo_mode and self.demo_broker is not None:
                        res = self.demo_broker.place_order(
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            price=price,
                            tag="phase26_ultra_multi",
                        )
                    else:
                        res = self.router.route_order(
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            order_type="MARKET",
                            tag="phase26_ultra_multi",
                            risk_ctx=risk_ctx,
                        )
            
                    log.info("[🧾 %d] %s result: %s", exec_idx, symbol, res)
            
                # IMPORTANT: once multi-symbol execution completes, exit tick
                return

            # -------------------------------------------------------------
            # Phase A — execution dispatch (single vs multi-symbol)
            # -------------------------------------------------------------

            if decisions:
                top = decisions[0]
                symbol = top.symbol
                price = float(top.alloc.debug.get("price", 0.0) if top.alloc and top.alloc.debug else 0.0) or float(
                    next((c["price"] for c in candidates if c.get("symbol") == symbol), 0.0)
                )
            
                fused_score_raw = float(top.fused_score)
                decision_score = float(top.ensemble_score)
            
                log.info(
                    "🎯 Allocator top symbol=%s | price=%.2f | fused=%.4f | decision=%.4f | tgt_qty=%s tgt_w=%.4f",
                    symbol,
                    float(price),
                    fused_score_raw,
                    decision_score,
                    getattr(top.alloc, "target_qty", None),
                    float(getattr(top.alloc, "target_weight", 0.0) or 0.0),
                )
            else:
                # Fallback: original best-pick logic (keeps bot alive if allocator returns none)
                best = max(
                    candidates,
                    key=lambda d: abs(d.get("ensemble_score", d["fused_score"])),
                )
                fused_score_raw = float(best["fused_score"])
                decision_score = float(best.get("ensemble_score", fused_score_raw))
                symbol = best["symbol"]
                price = float(best["price"])
            
                log.info(
                    "🎯 Best symbol=%s | price=%.2f | fused=%.4f | decision=%.4f | weights=%s",
                    symbol,
                    price,
                    fused_score_raw,
                    decision_score,
                    best.get("weights", {}),
                )
            
            
            # Keep your existing safety check
            if price <= 0:
                log.warning("Invalid price %.4f for %s; skipping.", price, symbol)
                return

            
            # ============================================================
            # Phase 27 MicroAllocator — Compute proper target sizing
            # ============================================================
            

            snapshot_before = self._portfolio_snapshot()
            positions = snapshot_before.get("positions", {}) or {}
            equity = float(snapshot_before.get("equity", 0.0) or 0.0)
            
            if equity <= 0.0:
                log.warning("Equity <= 0; skipping tick (allocator/risk protection).")
                return
            
            # Use realized vol from PortfolioBrain if available; fall back to snapshot field.
            try:
                vol_proxy = float(self.pbrain.get_realized_vol() or 1.0)
            except Exception:
                vol_proxy = float(snapshot_before.get("volatility", 1.0) or 1.0)
            
            # -------------------------------------------------------------
            # Phase 28 — MultiSymbolAllocator (portfolio-aware sizing)
            # -------------------------------------------------------------
            if getattr(self, "multi_alloc", None) is not None:
                try:
                    multi_decisions = self.multi_alloc.allocate(
                        equity=equity,
                        positions=positions,
                        candidates=candidates,
                        volatility_map=None,
                        mode=self.mode,
                        ctx={"tick_idx": ctx.get("tick_idx"), "volatility": vol_proxy},
                    )
                except Exception:
                    log.exception("Phase 28: MultiSymbolAllocator.allocate failed; falling back to single-symbol sizing.")
                    multi_decisions = []
            
                if not multi_decisions:
                    log.info("Phase 28: No symbols selected by MultiSymbolAllocator; skipping tick.")
                    return
            
                sel = multi_decisions[0]
                symbol = sel.symbol
                decision_score = float(sel.ensemble_score)
            
                # price from candidate map (keeps consistency)
                _price_map = {c["symbol"]: float(c.get("price", 0.0) or 0.0) for c in candidates}
                price = float(_price_map.get(symbol, 0.0) or 0.0)
            
                desired_shares = float(sel.alloc.target_qty)  # signed target position (shares)
                log.info(
                    "Phase 28 selection #%d %s: ensemble=%.4f fused=%.4f → tgt_weight=%.4f qty=%s side=%s",
                    sel.rank,
                    symbol,
                    float(sel.ensemble_score),
                    float(sel.fused_score),
                    float(sel.alloc.target_weight),
                    str(sel.alloc.target_qty),
                    str(sel.alloc.side),
                )
            else:
                # Fallback (Phase 26 legacy): derive desired shares from decision_score and max_equity_risk_pct
                symbol = best["symbol"]
                price = float(best["price"])
                cur_pos = positions.get(symbol, {"qty": 0.0, "price": price})
                cur_qty = float(cur_pos.get("qty", 0.0))
            
                max_alloc = equity * max_equity_risk_pct
                target_dollar = max_alloc * float(decision_score)
            
                if price <= 0:
                    log.warning("Invalid price %.4f for %s; skipping.", price, symbol)
                    return
            
                if abs(target_dollar) < 1.0:
                    log.info(
                        "Target dollar exposure |%.2f| < 1.0; treating as no-op (monitor-only).",
                        target_dollar,
                    )
                    return
            
                target_shares = target_dollar / price
                desired_shares = target_shares if decision_score > 0 else -target_shares
            
            # Current position (for the selected symbol)
            cur_pos = positions.get(symbol, {"qty": 0.0, "price": price})
            cur_qty = float(cur_pos.get("qty", 0.0))
            
            position_value = cur_qty * price
            log.info(
                "Current position %s: qty=%.4f value=%.2f",
                symbol,
                cur_qty,
                position_value,
            )
            
            # Persist intended positions (for safety / replay / diagnostics)
            intended_positions = {}
            for sym, pos in positions.items():
                intended_positions[sym] = {
                    "qty": float(pos.get("qty", 0.0)),
                    "price": float(pos.get("price", 0.0)),
                }
            
            intended_positions[symbol] = {
                "qty": float(desired_shares),
                "price": float(price),
            }
            
            self._save_bot_state(equity=equity, positions=intended_positions)
            
            diff_shares = desired_shares - cur_qty
            if abs(diff_shares) < 0.01 and not self.demo_mode:
                log.info(
                    "Diff shares %.4f too small; skipping trade (within noise).",
                    diff_shares,
                )
                return
            
            side = "BUY" if diff_shares > 0 else "SELL"
            order_qty = abs(diff_shares)
            cooldown_key = f"{symbol}_last_trade_ts"
            last_trade_ts = getattr(self, cooldown_key, 0.0)
            now_ts = time.time()
            if (now_ts - last_trade_ts) < symbol_cooldown_sec and not self.demo_mode:
                log.info(
                    "Symbol %s under cooldown (%.1fs remaining); skipping trade.",
                    symbol,
                    symbol_cooldown_sec - (now_ts - last_trade_ts),
                )
                return

            setattr(self, cooldown_key, now_ts)

            equity_before = equity
            positions_before = positions
            gross_before = 0.0
            for pos in positions_before.values():
                q = float(pos.get("qty", 0.0))
                px = float(pos.get("price", 0.0))
                gross_before += abs(q * px)

            log.info(
                "Pre-trade equity=%.2f gross_exposure=%.2f (before new order).",
                equity_before,
                gross_before,
            )

            if self.mode.upper() == "DEMO":
                final_qty = order_qty
            else:
                max_qty = gross_before * max_equity_risk_pct / max(price, 1e-6)
                max_qty = max(max_qty, 0.0)
                final_qty = min(order_qty, max_qty)
                if final_qty <= 0.0:
                    log.info(
                        "Risk envelope clamps order to 0 (max_qty=%.4f); skipping.",
                        max_qty,
                    )
                    return

            if self.mode.upper() == "DEMO":
                log.info(
                    "DEMO MODE: would place %s %s x %.4f @ %.2f (no real order).",
                    side,
                    symbol,
                    final_qty,
                    price,
                )
            else:
                log.info(
                    "Placing %s %s x %.4f @ %.2f",
                    side,
                    symbol,
                    final_qty,
                    price,
                )

            if self.guardrails_enabled:
                risk_ctx = self._portfolio_snapshot()
                if not self.risk_envelope.allow_order(
                    symbol=symbol,
                    side=side,
                    qty=final_qty,
                    price=price,
                    ctx=risk_ctx,
                ):
                    log.warning(
                        "RiskEnvelopeController blocked order %s %s x %.4f @ %.2f",
                        side,
                        symbol,
                        final_qty,
                        price,
                    )
                    return
            else:
                risk_ctx = {}

            try:
                # -------------------------------------------------------------------
                # OrderLedger stub (if ai.ledger.order_ledger does not exist)
                # -------------------------------------------------------------------
                try:
                    from ai.ledger.order_ledger import OrderLedger
                except Exception:
                    class OrderLedger:
                        def __init__(self, *args, **kwargs):
                            self.orders = []
                
                        def record(self, order_info: dict):
                            """Store order log entries in memory for now."""
                            self.orders.append(order_info)
                
                        def dump(self):
                            """Optional placeholder if you want to write to disk later."""
                            return self.orders
                
                    print("⚠️  Using OrderLedger stub (ai.ledger.order_ledger not found)")


                ledger = OrderLedger()
                ledger.log_order_submit(
                    symbol=symbol,
                    side=side,
                    qty=final_qty,
                    price=price,
                    tag="phase26_ultra",
                    equity=equity_before,
                    gross_exposure=gross_before,
                    meta={
                        "aggression": float(self._aggression_factor),
                        "fused_score": float(fused_score_raw),
                        "decision_score": float(decision_score),
                    },
                )
            except Exception:
                log.exception("OrderLedger: failed to log order_submit.")

            had_trade = True

            # Route via SmartOrderRouter, DemoBroker, or Replay (Phase G)
            if getattr(self, "_disable_live_execution", False):
                if not hasattr(self, "_replay_fills"):
                    self._replay_fills = []
                self._replay_fills.append({
                    "trace": ctx.get("trace_id"),
                    "symbol": symbol,
                    "side": side,
                    "qty": float(final_qty),
                    "price": float(price),
                })
                res = {"ok": True, "id": f"replay-{len(self._replay_fills)}", "replay": True}
            elif self.demo_mode and self.demo_broker is not None:
                res = self.demo_broker.place_order(
                    symbol=symbol,
                    side=side,
                    qty=final_qty,
                    price=price,
                    tag="phase26_ultra",
                )
            else:
                res = self.router.route_order(
                    symbol=symbol,
                    side=side,
                    qty=final_qty,
                    order_type="MARKET",
                    tag="phase26_ultra",
                    risk_ctx=risk_ctx,
                )

            log.info("🧾 RouteRes: %s", res)

            snapshot_after = self._portfolio_snapshot()
            positions_after = snapshot_after.get("positions", {})
            equity_after = snapshot_after.get("equity", 0.0)

            gross_exposure = 0.0
            for pos in positions_after.values():
                q = float(pos.get("qty", 0.0))
                px = float(pos.get("price", 0.0))
                gross_exposure += abs(q * px)

            fill = res.get("fill") or {}
            pnl = 0.0
            trade_cost = 0.0

            if fill:
                pnl = float(fill.get("pnl", 0.0))
                trade_cost = float(fill.get("commission", 0.0))

            trade_info = {
                "symbol": symbol,
                "side": side,
                "qty": float(final_qty),
                "price": float(price),
                "pnl": pnl,
                "commission": trade_cost,
                "equity_before": equity_before,
                "equity_after": float(equity_after),
                "gross_exposure": gross_exposure,
            }

            if not candidates:
                log.warning("No prices available for any symbol; skipping tick.")
                return

            # -------------------------------------------------------------
            # Phase 28: Multi-symbol allocation
            # -------------------------------------------------------------
            # Build volatility map if vol_engine is available
            vol_map: Dict[str, float] = {}
            try:
                if hasattr(self, "vol_engine") and self.vol_engine is not None:
                    for c in candidates:
                        sym = c["symbol"]
                        vol_map[sym] = float(self.vol_engine.get_vol(sym) or 1.0)
                else:
                    base_vol = float(self.pbrain.get_realized_vol() or 1.0)
                    for c in candidates:
                        vol_map[c["symbol"]] = base_vol
            except Exception:
                log.exception("Phase 28: failed to build volatility map; defaulting to 1.0.")
                for c in candidates:
                    vol_map[c["symbol"]] = 1.0

            # Apply entry threshold at per-symbol level (skip weak signals)
            tradable_candidates: list[Dict[str, Any]] = []
            for c in candidates:
                fused_score_raw = float(c.get("fused_score", 0.0))
                decision_score = float(c.get("ensemble_score", fused_score_raw))
                if not self.demo_mode and abs(decision_score) < entry_threshold:
                    log.info(
                        "Symbol %s decision score %.4f below threshold %.4f; skipping.",
                        c["symbol"],
                        decision_score,
                        entry_threshold,
                    )
                    continue
                tradable_candidates.append(c)

            if not tradable_candidates:
                log.info(
                    "All symbols below decision threshold %.4f; no trades this tick.",
                    entry_threshold,
                )
                return

            # Snapshot once at the beginning of allocations
            snapshot_before = self._portfolio_snapshot()
            positions = snapshot_before.get("positions", {})
            equity = float(snapshot_before.get("equity", 0.0))

            # Run portfolio-aware allocator
            multi_decisions = self.multi_alloc.allocate(
                equity=equity,
                positions=positions,
                candidates=tradable_candidates,
                volatility_map=vol_map,
                mode=self.mode,
                ctx={"aggression": float(self._aggression_factor)},
            )

            if not multi_decisions:
                log.info("Phase 28: MultiSymbolAllocator returned no trade decisions.")
                return

            # -------------------------------------------------------------
            # Execute each allocator decision (per symbol)
            # -------------------------------------------------------------
            for decision_idx, d in enumerate(multi_decisions, start=1):
                symbol = d.symbol
                alloc_decision = d.alloc
                fused_score_raw = float(d.fused_score)
                decision_score = float(d.ensemble_score)

                price = None
                for c in candidates:
                    if c["symbol"] == symbol:
                        price = float(c["price"])
                        break
                if price is None or price <= 0:
                    log.warning("Phase 28: invalid price for %s; skipping.", symbol)
                    continue

                # Current position info
                cur_pos = positions.get(symbol, {"qty": 0.0, "price": price})
                cur_qty = float(cur_pos.get("qty", 0.0))
                position_value = cur_qty * price

                log.info(
                    "[%d] Current position %s: qty=%.4f value=%.2f | fused=%.4f decision=%.4f",
                    decision_idx,
                    symbol,
                    cur_qty,
                    position_value,
                    fused_score_raw,
                    decision_score,
                )

                # Allocator gives us target_qty and weights
                desired_shares = float(alloc_decision.target_qty)
                diff_shares = desired_shares - cur_qty

                if abs(diff_shares) < 0.01 and not self.demo_mode:
                    log.info(
                        "[%d] %s diff shares %.4f too small; skipping trade (within noise).",
                        decision_idx,
                        symbol,
                        diff_shares,
                    )
                    continue

                side = alloc_decision.side or ("BUY" if diff_shares > 0 else "SELL")
                order_qty = abs(diff_shares)

                # Per-symbol cooldown
                cooldown_key = f"{symbol}_last_trade_ts"
                last_trade_ts = getattr(self, cooldown_key, 0.0)
                now_ts = time.time()
                if (now_ts - last_trade_ts) < symbol_cooldown_sec and not self.demo_mode:
                    log.info(
                        "[%d] Symbol %s under cooldown (%.1fs remaining); skipping trade.",
                        decision_idx,
                        symbol,
                        symbol_cooldown_sec - (now_ts - last_trade_ts),
                    )
                    continue

                setattr(self, cooldown_key, now_ts)

                # Pre-trade exposure snapshot (per symbol)
                equity_before = equity
                positions_before = positions
                gross_before = 0.0
                for pos in positions_before.values():
                    q = float(pos.get("qty", 0.0))
                    px = float(pos.get("price", 0.0))
                    gross_before += abs(q * px)

                log.info(
                    "[%d] Pre-trade equity=%.2f gross_exposure=%.2f (before new order for %s).",
                    decision_idx,
                    equity_before,
                    gross_before,
                    symbol,
                )

                # Additional equity risk clamp (on top of allocator's weight)
                if self.mode.upper() == "DEMO":
                    final_qty = order_qty
                else:
                    max_qty = gross_before * max_equity_risk_pct / max(price, 1e-6)
                    max_qty = max(max_qty, 0.0)
                    final_qty = min(order_qty, max_qty)
                    if final_qty <= 0.0:
                        log.info(
                            "[%d] Risk envelope pre-clamp: order %s %s reduces to 0 (max_qty=%.4f); skipping.",  # noqa: E501
                            decision_idx,
                            side,
                            symbol,
                            max_qty,
                        )
                        continue

                if self.mode.upper() == "DEMO":
                    log.info(
                        "[%d] DEMO MODE: would place %s %s x %.4f @ %.2f (no real order).",
                        decision_idx,
                        side,
                        symbol,
                        final_qty,
                        price,
                    )
                else:
                    log.info(
                        "[%d] Placing %s %s x %.4f @ %.2f",
                        decision_idx,
                        side,
                        symbol,
                        final_qty,
                        price,
                    )

                # RiskEnvelopeController check
                if self.guardrails_enabled:
                    risk_ctx = self._portfolio_snapshot()
                    if not self.risk_envelope.allow_order(
                        symbol=symbol,
                        side=side,
                        qty=final_qty,
                        price=price,
                        ctx=risk_ctx,
                    ):
                        log.warning(
                            "[%d] RiskEnvelopeController blocked order %s %s x %.4f @ %.2f",
                            decision_idx,
                            side,
                            symbol,
                            final_qty,
                            price,
                        )
                        continue
                else:
                    risk_ctx = {}

                # -----------------------------------------
                # Route the order (DemoBroker or Router)
                # -----------------------------------------
                had_trade = True

                if self.demo_mode and self.demo_broker is not None:
                    res = self.demo_broker.place_order(
                        symbol=symbol,
                        side=side,
                        qty=final_qty,
                        price=price,
                        tag="phase26_ultra",
                    )
                else:
                    res = self.router.route_order(
                        symbol=symbol,
                        side=side,
                        qty=final_qty,
                        order_type="MARKET",
                        tag="phase26_ultra",
                        risk_ctx=risk_ctx,
                    )

                log.info("[🧾 %d] RouteRes for %s: %s", decision_idx, symbol, res)

                # -----------------------------------------
                # Post-trade: snapshot, PnL, logging, reward
                # -----------------------------------------
                snapshot_after = self._portfolio_snapshot()
                positions_after = snapshot_after.get("positions", {})
                equity_after = snapshot_after.get("equity", 0.0)

                gross_exposure = 0.0
                for pos in positions_after.values():
                    q = float(pos.get("qty", 0.0))
                    px = float(pos.get("price", 0.0))
                    gross_exposure += abs(q * px)

                fill = res.get("fill") or {}
                pnl = 0.0
                trade_cost = 0.0

                if fill:
                    pnl = float(fill.get("pnl", 0.0))
                    trade_cost = float(fill.get("commission", 0.0))

                trade_info = {
                    "symbol": symbol,
                    "side": side,
                    "qty": float(final_qty),
                    "price": float(price),
                    "pnl": pnl,
                    "commission": trade_cost,
                    "equity_before": equity_before,
                    "equity_after": float(equity_after),
                    "gross_exposure": gross_exposure,
                }

                # OrderLedger logging
                try:
                    try:
                        from ai.ledger.order_ledger import OrderLedger
                    except Exception:
                        class OrderLedger:
                            def __init__(self, *args, **kwargs):
                                self.orders = []

                            def record(self, order_info: dict):
                                self.orders.append(order_info)

                            def dump(self):
                                return self.orders

                        print("⚠️  Using OrderLedger stub (ai.ledger.order_ledger not found)")

                    ledger = OrderLedger()
                    ledger.log_order_result(
                        symbol=symbol,
                        side=side,
                        qty=final_qty,
                        price=price,
                        tag="phase26_ultra",
                        result=res,
                        meta=trade_info,
                    )
                except Exception:
                    log.exception("OrderLedger: failed to log order_result.")

                # Reward & performance recording
                reward = self.reward_engine.compute_reward({"info": trade_info})
                try:
                    perf_rec = PerformanceRecorder()
                    perf_rec.record(
                        policy_name=(
                            self.fusion_brain.last_policy_used
                            if hasattr(self.fusion_brain, "last_policy_used")
                            else "UnknownPolicy"
                        ),
                        reward=float(reward),
                        win_flag=1 if float(reward) > 0 else 0,
                    )
                except Exception as e:
                    log.error(f"Perf logging error in Phase 26: {e}")

                # Replay logging
                obs_vec = [price, float(self._aggression_factor), float(decision_score)]
                next_obs_vec = obs_vec
                replay_row = {
                    "obs": obs_vec,
                    "next_obs": next_obs_vec,
                    "reward": float(reward),
                    "done": False,
                    "info": trade_info,
                }
                try:
                    with self.replay_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(replay_row) + "\n")
                except Exception:
                    log.exception("Failed to append to Phase 26 replay file.")

                # Fusion engine online-learning hook
                if hasattr(self, "fusion_engine") and self.fusion_engine is not None:
                    try:
                        self.fusion_engine.update_performance(
                            symbol=symbol,
                            reward=float(reward),
                            info=trade_info,
                        )
                    except Exception:
                        log.exception("FusionEngine.update_performance failed.")

                # StabilityGuardian update
                try:
                    if self.guardrails_enabled:
                        self.stability_guardian.update(
                            equity=float(equity_after),
                            pnl=pnl,
                            gross_exposure=gross_exposure,
                        )
                except Exception:
                    log.exception("StabilityGuardian update failed.")

                # Regime detection (optional)
                try:
                    if getattr(self, "regime_detector", None) is not None:
                        price_series = [c["price"] for c in candidates]
                        snapshot_reg = self._portfolio_snapshot()
                        regime = self.regime_detector.detect(
                            prices=price_series,
                            portfolio_snapshot=snapshot_reg,
                        )
                        self._current_regime = regime
                        log.info(
                            "🌐 Regime: %s | risk=%s | score=%.3f",
                            regime.name,
                            regime.risk_level,
                            regime.score,
                        )
                except Exception:
                    log.exception("Phase120: regime detection failed in Phase 26 loop.")

            try:
                # -------------------------------------------------------------------
                # OrderLedger stub (if ai.ledger.order_ledger does not exist)
                # -------------------------------------------------------------------
                try:
                    from ai.ledger.order_ledger import OrderLedger
                except Exception:
                    class OrderLedger:
                        def __init__(self, *args, **kwargs):
                            self.orders = []
                
                        def record(self, order_info: dict):
                            """Store order log entries in memory for now."""
                            self.orders.append(order_info)
                
                        def dump(self):
                            """Optional placeholder if you want to write to disk later."""
                            return self.orders
                
                    print("⚠️  Using OrderLedger stub (ai.ledger.order_ledger not found)")


                ledger = OrderLedger()
                ledger.log_order_result(
                    symbol=symbol,
                    side=side,
                    qty=final_qty,
                    price=price,
                    tag="phase26_ultra",
                    result=res,
                    meta=trade_info,
                )
            except Exception:
                log.exception("OrderLedger: failed to log order_result.")

            reward = self.reward_engine.compute_reward({"info": trade_info})

            try:
                perf_rec = PerformanceRecorder()
                perf_rec.record(
                    policy_name=(
                        self.fusion_brain.last_policy_used
                        if hasattr(self.fusion_brain, "last_policy_used")
                        else "UnknownPolicy"
                    ),
                    reward=float(reward),
                    win_flag=1 if float(reward) > 0 else 0,
                )
            except Exception as e:
                log.error(f"Perf logging error in Phase 26: {e}")

            obs_vec = [price, float(self._aggression_factor), float(decision_score)]
            next_obs_vec = obs_vec

            replay_row = {
                "obs": obs_vec,
                "next_obs": next_obs_vec,
                "reward": float(reward),
                "done": False,
                "info": trade_info,
            }
            try:
                with self.replay_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(replay_row) + "\n")
            except Exception:
                log.exception("Failed to append to Phase 26 replay file.")

            if hasattr(self, "fusion_engine") and self.fusion_engine is not None:
                try:
                    self.fusion_engine.update_performance(
                        symbol=symbol,
                        reward=float(reward),
                        info=trade_info,
                    )
                except Exception:
                    log.exception("FusionEngine.update_performance failed.")

            try:
                if self.guardrails_enabled:
                    self.stability_guardian.update(
                        equity=float(equity_after),
                        pnl=pnl,
                        gross_exposure=gross_exposure,
                    )
            except Exception:
                log.exception("StabilityGuardian update failed.")

            try:
                if getattr(self, "regime_detector", None) is not None:
                    price_series = [c["price"] for c in candidates]
                    snapshot_reg = self._portfolio_snapshot()
                    regime = self.regime_detector.detect(
                        prices=price_series,
                        portfolio_snapshot=snapshot_reg,
                    )
                    self._current_regime = regime
                    log.info(
                        "🌐 Regime: %s | risk=%s | score=%.3f",
                        regime.name,
                        regime.risk_level,
                        regime.score,
                    )
            except Exception:
                log.exception("Phase120: regime detection failed in Phase 26 loop.")

        except Exception as e:
            tick_error = True
            self._error_ticks += 1
            log.error("💥 Tick error: %s", e, exc_info=True)
            try:
                record_error("phase26_tick", "exception", str(e))
            except Exception:
                pass
        finally:
            # Phase 114 – update counters and evaluate health
            try:
                # If no trade this tick, increment idle counter
                if not had_trade:
                    self._ticks_since_trade += 1
                else:
                    self._ticks_since_trade = 0

                # Simple decay: if this tick had no error, slowly reduce error counter
                if not tick_error and self._error_ticks > 0:
                    self._error_ticks -= 1

                # Option D – DEMO-mode PnL curve logging
                try:
                    if (
                        getattr(self, "demo_mode", False)
                        and getattr(self, "demo_broker", None) is not None
                        and getattr(self, "pnl_logger", None) is not None
                    ):
                        equity, unreal, realized = self.demo_broker.mark_to_market()
                        self.pnl_logger.maybe_log(
                            equity=equity,
                            cash=self.demo_broker.cash,
                            unrealized=unreal,
                            realized=realized,
                        )
                except Exception:
                    log.exception("PnL logging failed in Phase 26 tick.")

                # Evaluate execution health and act if needed
                if self.health_monitor_enabled:
                    decision = self.health_monitor.check(
                        error_ticks=self._error_ticks,
                        ticks_since_trade=self._ticks_since_trade,
                    )

                    if not decision.healthy and decision.reason:
                        log.error(
                            "Phase 114: Execution unhealthy → %s", decision.reason
                        )

                        # Option A – flatten all positions, stop trading, raise kill-switch
                        if decision.should_flatten:
                            try:
                                snapshot = self._portfolio_snapshot()
                                self._flatten_positions_via_router(snapshot)
                            except Exception:
                                log.exception(
                                    "Phase 114: flatten via router failed despite unhealthy execution."  # noqa: E501
                                )

                        if decision.raise_kill_switch:
                            try:
                                self.kill_flag_path.write_text(
                                    "Execution health monitor triggered kill-switch.\n",  # noqa: E501
                                    encoding="utf-8",
                                )
                                log.error(
                                    "Phase 114: kill-switch file created at %s",
                                    self.kill_flag_path,
                                )
                            except Exception:
                                log.exception(
                                    "Phase 114: failed to create kill-switch file."
                                )

                        self.running = False

            except Exception:
                log.exception("Phase 114: health monitoring update failed.")

            elapsed = time.time() - start_ts
            log.info("✅ Tick completed in %.3fs", elapsed)

    # =================================================================
    # Bot state saving (Phase 112)
    # =================================================================
    def _save_bot_state(self, equity: float, positions: Dict[str, Dict[str, float]]) -> None:  # noqa: E501
        """
        Save bot's internal view of positions for Phase 112 reconciliation.

        positions format:
            { "AAPL": {"qty": 10.0, "price": 190.12}, ... }
        """
        self._bot_state["equity"] = float(equity)
        self._bot_state["positions"] = positions
        self._bot_state["updated_at"] = time.time()
        try:
            with self._bot_state_path.open("w", encoding="utf-8") as f:
                json.dump(self._bot_state, f, indent=2)
        except Exception:
            log.exception("Failed to persist Phase 26 bot state.")

    # =================================================================
    # Run Loop
    # =================================================================
    def run(self) -> None:
        """Run Phase 26 via AdaptiveAutoScalingExecutor."""
        self.autoscaler.run(
            tick_fn=self._tick,
            state_fn=self._autoscale_state,
            running_flag_fn=lambda: self.running,
        )

        # Option D – on graceful shutdown, try exporting a DEMO PnL PNG
        try:
            if (
                getattr(self, "demo_mode", False)
                and getattr(self, "pnl_logger", None) is not None
            ):
                self.pnl_logger.export_png_if_possible()
        except Exception:
            log.exception("PnLCurveLogger export failed on shutdown.")


# Import-safe compatibility alias (used by Phase-G replay runners)
Phase26RealtimeUltra = RealTimeExecutionLoop


def main(argv=None):
    ensure_env_loaded()

    max_restarts = int(os.getenv("PHASE26_AUTORESTART_MAX", "50"))
    backoff_sec = float(os.getenv("PHASE26_AUTORESTART_BACKOFF_SEC", "3.0"))
    restart_count = 0

    while True:
        try:
            loop = RealTimeExecutionLoop()
            loop.run()
            # If loop exits cleanly (running flag false), stop restarting.
            log.warning("Phase26 loop exited cleanly; stopping.")
            return

        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt: shutting down.")
            return

        except Exception as e:
            restart_count += 1
            log.error("FATAL loop crash #%d: %s", restart_count, e, exc_info=True)

            if restart_count >= max_restarts:
                log.error("Max restarts reached (%d). Exiting.", max_restarts)
                raise

            time.sleep(backoff_sec)



if __name__ == "__main__":
    main(sys.argv[1:])
