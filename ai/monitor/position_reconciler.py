# ai/monitor/position_reconciler.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ai.utils.alpaca_client import AlpacaClient
from ai.execution.smart_order_router import SmartOrderRouter

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationConfig:
    # Core paths
    internal_state_path: str = "data/runtime/phase26_bot_state.json"
    report_dir: str = "data/reports"

    # Matching tolerances
    tolerance_shares: float = 0.01          # <= this is considered equal (shares)
    tolerance_notional_pct: float = 0.5     # % of equity diff considered OK

    # Heal / auto-fix modes
    heal_mode: str = "none"                 # "none" | "align_internal" | "close_strays"
    use_router_for_heal: bool = True        # if False, heal via direct broker API

    # Advanced drift safety
    drift_killswitch_threshold_pct: float = 3.0   # trigger kill-switch if total drift >= this %
    drift_symbol_threshold: int = 2               # kill if mismatches on >= this many symbols

    # Mild-drift auto-alignment
    auto_heal_notional_pct: float = 1.0           # <= this % drift → allow auto-align
    auto_align_internal: bool = True              # auto-align internal view for mild drift


@dataclass
class SymbolDiff:
    symbol: str
    qty_internal: float
    qty_broker: float
    price_broker: float
    notional_internal: float
    notional_broker: float
    diff_qty: float
    diff_notional: float
    status: str      # MATCH / MISMATCH / ONLY_INTERNAL / ONLY_BROKER


@dataclass
class ReconciliationResult:
    timestamp: str
    equity_internal: float
    equity_broker: float
    diffs: List[SymbolDiff]
    summary: Dict[str, Any]
    heal_actions: List[Dict[str, Any]]


class PositionReconciler:
    """
    Phase 112 – Trade / Position Reconciliation (advanced)

    Compares:
        • Bot's internal "view" of positions (from internal_state_path)
        • Live broker (Alpaca) positions

    Outputs:
        • JSON + CSV reports in report_dir
        • Optional healing (align internal view or close stray broker positions)
        • Drift severity metrics and automatic kill-switch file when severe
    """

    def __init__(self, cfg: ReconciliationConfig) -> None:
        self.cfg = cfg
        self.internal_state_path = Path(cfg.internal_state_path)
        self.report_dir = Path(cfg.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Broker + router
        self.broker = AlpacaClient()

        self.router: SmartOrderRouter | None = None
        if self.cfg.use_router_for_heal:
            try:
                self.router = SmartOrderRouter(
                    risk_cfg_path="configs/phase69c_risk_envelope.yaml",
                    multix_cfg_path="configs/phase69d_multix.yaml",
                    portfolio_provider=self._portfolio_snapshot,
                    primary_broker=self.broker,
                )
            except Exception:
                logger.exception(
                    "PositionReconciler: failed to init SmartOrderRouter; heal will fallback to broker."
                )

    # ------------------------------------------------------------------
    # Public: Run one reconciliation pass
    # ------------------------------------------------------------------
    def run_once(self) -> ReconciliationResult:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        internal_positions, equity_internal = self._load_internal_state()
        broker_positions, equity_broker = self._load_broker_state()

        diffs, severity_pct = self._compute_diffs(
            internal_positions=internal_positions,
            broker_positions=broker_positions,
            equity_broker=equity_broker,
        )

        summary = self._summarize(diffs=diffs)
        summary["severity_pct"] = severity_pct
        summary["kill_switch_triggered"] = False

        heal_actions: List[Dict[str, Any]] = []

        # Optional healing / alignment
        if self.cfg.heal_mode != "none" or self.cfg.auto_align_internal:
            try:
                heal_actions = self._heal(
                    diffs=diffs,
                    internal_positions=internal_positions,
                    broker_positions=broker_positions,
                    equity_broker=equity_broker,
                )
            except Exception:
                logger.exception("PositionReconciler: heal step failed.")

        # Severe drift → write kill-switch flag (Phase 26 will see this)
        if self._should_trigger_killswitch(summary):
            kill_msg = (
                "🚨 PHASE112 HARD DRIFT DETECTED\n"
                f"Severity: {severity_pct:.2f}% (threshold={self.cfg.drift_killswitch_threshold_pct}%)\n"
                f"Mismatch symbols: {summary.get('mismatch_count')} "
                f"(threshold={self.cfg.drift_symbol_threshold})\n"
                "Trading must stop. Investigate reconciliation reports."
            )
            logger.error(kill_msg)

            killflag = Path("data/runtime/trading_disabled.flag")
            killflag.parent.mkdir(parents=True, exist_ok=True)
            killflag.write_text(kill_msg, encoding="utf-8")

            try:
                from tools.telegram_alerts import notify
                notify(kill_msg, kind="guardian", meta={"phase": 112})
            except Exception:
                logger.exception("PositionReconciler: failed to send Telegram kill-switch alert.")

            summary["kill_switch_triggered"] = True

        result = ReconciliationResult(
            timestamp=ts,
            equity_internal=equity_internal,
            equity_broker=equity_broker,
            diffs=diffs,
            summary=summary,
            heal_actions=heal_actions,
        )

        self._write_reports(ts, result)
        logger.info(
            "Phase 112 Reconciliation complete (matches=%s, mismatches=%s, only_broker=%s, only_internal=%s, severity=%.2f%%)",
            summary.get("match_count"),
            summary.get("mismatch_count"),
            summary.get("only_broker_count"),
            summary.get("only_internal_count"),
            severity_pct,
        )

        return result

    # ------------------------------------------------------------------
    # Internal state loader (bot's view)
    # ------------------------------------------------------------------
    def _load_internal_state(self) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        Expected format in JSON:
            {
              "equity": 12345.67,
              "positions": {
                  "AAPL": {"qty": 10.0, "price": 190.12},
                  "MSFT": {"qty": -5.0, "price": 320.10}
              },
              ...
            }
        """
        if not self.internal_state_path.exists():
            logger.warning(
                "Internal state file %s not found; using empty state.",
                self.internal_state_path,
            )
            return {}, 0.0

        try:
            with self.internal_state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception(
                "PositionReconciler: failed to load internal state from %s; treating as empty.",
                self.internal_state_path,
            )
            return {}, 0.0

        positions = data.get("positions", {}) or {}
        equity = float(data.get("equity", 0.0))

        norm_positions: Dict[str, Dict[str, float]] = {}
        for sym, p in positions.items():
            try:
                qty = float(p.get("qty", 0.0))
                price = float(p.get("price", 0.0))
                norm_positions[sym] = {"qty": qty, "price": price}
            except Exception:
                logger.exception(
                    "PositionReconciler: failed to normalize internal pos %r", p
                )

        return norm_positions, equity

    # ------------------------------------------------------------------
    # Broker state loader (Alpaca view)
    # ------------------------------------------------------------------
    def _load_broker_state(self) -> Tuple[Dict[str, Dict[str, float]], float]:
        acct = self.broker.get_account()
        equity = float(getattr(acct, "equity", 0.0))

        api_positions = self.broker.get_positions()
        positions: Dict[str, Dict[str, float]] = {}
        for p in api_positions:
            try:
                sym = getattr(p, "symbol", None)
                qty_raw = getattr(p, "qty", getattr(p, "quantity", 0.0))
                price_raw = getattr(
                    p,
                    "current_price",
                    getattr(p, "avg_entry_price", 0.0),
                )
                if sym is None:
                    continue
                qty = float(qty_raw)
                price = float(price_raw)
                positions[str(sym)] = {"qty": qty, "price": price}
            except Exception:
                logger.exception(
                    "PositionReconciler: failed to normalize broker position %r", p
                )

        return positions, equity

    # ------------------------------------------------------------------
    # Diff computation + severity
    # ------------------------------------------------------------------
    def _compute_diffs(
        self,
        internal_positions: Dict[str, Dict[str, float]],
        broker_positions: Dict[str, Dict[str, float]],
        equity_broker: float,
    ) -> Tuple[List[SymbolDiff], float]:
        symbols = set(internal_positions.keys()) | set(broker_positions.keys())
        diffs: List[SymbolDiff] = []

        equity_safe = max(equity_broker, 1e-9)

        for sym in sorted(symbols):
            i_pos = internal_positions.get(sym)
            b_pos = broker_positions.get(sym)

            qty_i = float(i_pos["qty"]) if i_pos is not None else 0.0
            price_i = float(i_pos["price"]) if i_pos is not None else 0.0
            qty_b = float(b_pos["qty"]) if b_pos is not None else 0.0
            price_b = float(b_pos["price"]) if b_pos is not None else price_i

            notional_i = qty_i * price_b
            notional_b = qty_b * price_b

            diff_qty = qty_b - qty_i
            diff_notional = notional_b - notional_i

            if i_pos is None and b_pos is not None:
                status = "ONLY_BROKER"
            elif i_pos is not None and b_pos is None:
                status = "ONLY_INTERNAL"
            else:
                # Both exist – compare with tolerances
                if abs(diff_qty) <= self.cfg.tolerance_shares:
                    pct = abs(diff_notional) / equity_safe * 100.0
                    if pct <= self.cfg.tolerance_notional_pct:
                        status = "MATCH"
                    else:
                        status = "MISMATCH"
                else:
                    status = "MISMATCH"

            diffs.append(
                SymbolDiff(
                    symbol=sym,
                    qty_internal=qty_i,
                    qty_broker=qty_b,
                    price_broker=price_b,
                    notional_internal=notional_i,
                    notional_broker=notional_b,
                    diff_qty=diff_qty,
                    diff_notional=diff_notional,
                    status=status,
                )
            )

        # Total absolute notional drift as % of broker equity
        total_notional_drift = sum(abs(d.diff_notional) for d in diffs)
        severity_pct = total_notional_drift / equity_safe * 100.0

        return diffs, severity_pct

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(diffs: List[SymbolDiff]) -> Dict[str, Any]:
        match = sum(1 for d in diffs if d.status == "MATCH")
        mismatch = sum(1 for d in diffs if d.status == "MISMATCH")
        only_broker = sum(1 for d in diffs if d.status == "ONLY_BROKER")
        only_internal = sum(1 for d in diffs if d.status == "ONLY_INTERNAL")

        return {
            "match_count": match,
            "mismatch_count": mismatch,
            "only_broker_count": only_broker,
            "only_internal_count": only_internal,
            "total_symbols": len(diffs),
        }

    # ------------------------------------------------------------------
    # Advanced drift kill-switch condition
    # ------------------------------------------------------------------
    def _should_trigger_killswitch(self, summary: Dict[str, Any]) -> bool:
        severity = float(summary.get("severity_pct", 0.0))
        mismatches = int(summary.get("mismatch_count", 0))
        only_broker = int(summary.get("only_broker_count", 0))

        if severity >= self.cfg.drift_killswitch_threshold_pct:
            return True

        if (mismatches + only_broker) >= self.cfg.drift_symbol_threshold:
            return True

        return False

    # ------------------------------------------------------------------
    # Heal logic (advanced)
    # ------------------------------------------------------------------
    def _heal(
        self,
        diffs: List[SymbolDiff],
        internal_positions: Dict[str, Dict[str, float]],
        broker_positions: Dict[str, Dict[str, float]],
        equity_broker: float,
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        # Explicit heal_mode override – always align_internal
        if self.cfg.heal_mode == "align_internal":
            self._align_internal_to_broker(broker_positions, equity_broker)
            actions.append(
                {
                    "type": "align_internal",
                    "detail": "Internal state overwritten to match broker positions.",
                }
            )
            return actions

        # Close stray broker positions / big mismatches
        if self.cfg.heal_mode == "close_strays":
            stray_diffs = [
                d for d in diffs if d.status in ("ONLY_BROKER", "MISMATCH")
            ]
            for d in stray_diffs:
                if d.qty_broker == 0.0:
                    continue

                try:
                    side = "SELL" if d.qty_broker > 0 else "BUY"
                    qty = abs(d.qty_broker)

                    if self.router is not None and self.cfg.use_router_for_heal:
                        self.router.route_order(
                            symbol=d.symbol,
                            side=side,
                            qty=qty,
                            order_type="MARKET",
                            tag="phase112_heal",
                            risk_ctx=self._portfolio_snapshot(),
                        )
                    else:
                        self.broker.submit_order(
                            symbol=d.symbol,
                            side=side,
                            qty=qty,
                            order_type="market",
                            time_in_force="day",
                        )

                    actions.append(
                        {
                            "type": "close_stray",
                            "symbol": d.symbol,
                            "side": side,
                            "qty": qty,
                            "reason": d.status,
                        }
                    )
                    logger.warning(
                        "PositionReconciler: heal close_stray %s %s (%s)",
                        qty,
                        d.symbol,
                        d.status,
                    )
                except Exception:
                    logger.exception(
                        "PositionReconciler: failed to heal %s", d.symbol
                    )

        # Mild-drift auto-alignment (soft safety)
        if (
            self.cfg.auto_align_internal
            and self.cfg.auto_heal_notional_pct > 0.0
            and equity_broker > 0.0
        ):
            mild = [
                d
                for d in diffs
                if d.status == "MISMATCH"
                and abs(d.diff_notional) / equity_broker * 100.0
                <= self.cfg.auto_heal_notional_pct
            ]
            if mild:
                self._align_internal_to_broker(broker_positions, equity_broker)
                actions.append(
                    {
                        "type": "auto_align_mild",
                        "num_symbols": len(mild),
                        "reason": "small drift auto-healed internally",
                    }
                )
                logger.warning(
                    "PositionReconciler: auto-align internal state for mild drift (%d symbols).",
                    len(mild),
                )

        return actions

    def _align_internal_to_broker(
        self,
        broker_positions: Dict[str, Dict[str, float]],
        equity_broker: float,
    ) -> None:
        """
        Overwrite internal state JSON to match broker exactly.
        """
        state = {
            "equity": float(equity_broker),
            "positions": broker_positions,
            "updated_at": datetime.utcnow().isoformat(),
            "source": "phase112_align_internal_to_broker",
        }

        self.internal_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.internal_state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger.warning(
            "PositionReconciler: internal bot state overwritten with broker positions (%d symbols).",
            len(broker_positions),
        )

    # ------------------------------------------------------------------
    # Portfolio snapshot helper (for router risk_ctx)
    # ------------------------------------------------------------------
    def _portfolio_snapshot(self) -> Dict[str, Any]:
        acct = self.broker.get_account()
        equity = float(getattr(acct, "equity", 0.0))
        positions = self.broker.get_positions()

        pos_map: Dict[str, Dict[str, float]] = {}
        for p in positions:
            try:
                sym = getattr(p, "symbol", None)
                qty = float(getattr(p, "qty", 0.0))
                price = float(getattr(p, "current_price", 0.0))
                if sym:
                    pos_map[str(sym)] = {"qty": qty, "price": price}
            except Exception:
                logger.exception(
                    "PositionReconciler: snapshot normalize failed for %r", p
                )

        return {"equity": equity, "positions": pos_map}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _write_reports(self, ts: str, result: ReconciliationResult) -> None:
        json_path = self.report_dir / f"phase112_reconciliation_{ts}.json"
        csv_path = self.report_dir / f"phase112_reconciliation_{ts}.csv"

        payload = {
            "timestamp": result.timestamp,
            "equity_internal": result.equity_internal,
            "equity_broker": result.equity_broker,
            "summary": result.summary,
            "diffs": [asdict(d) for d in result.diffs],
            "heal_actions": result.heal_actions,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "symbol",
                    "qty_internal",
                    "qty_broker",
                    "price_broker",
                    "notional_internal",
                    "notional_broker",
                    "diff_qty",
                    "diff_notional",
                    "status",
                ]
            )
            for d in result.diffs:
                writer.writerow(
                    [
                        d.symbol,
                        d.qty_internal,
                        d.qty_broker,
                        d.price_broker,
                        d.notional_internal,
                        d.notional_broker,
                        d.diff_qty,
                        d.diff_notional,
                        d.status,
                    ]
                )

        logger.info("PositionReconciler: wrote JSON report → %s", json_path)
        logger.info("PositionReconciler: wrote CSV report  → %s", csv_path)
