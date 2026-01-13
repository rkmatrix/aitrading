# ai/memory/model_skill_memory.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SkillStats:
    # counts
    trades: int = 0
    wins: int = 0
    losses: int = 0

    # pnl
    pnl_net: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0  # negative (stored as negative)

    # equity curve for drawdown on realized pnl
    equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0  # positive number

    # variance proxy (for sharpe-ish)
    mean_r: float = 0.0
    m2_r: float = 0.0  # sum of squares of differences from mean (Welford)

    # confidence calibration (Brier score)
    brier_mean: float = 0.0
    brier_m2: float = 0.0  # for variance if needed
    brier_n: int = 0

    last_ts: float = 0.0


class ModelSkillMemory:
    """
    Phase D-3: Persistent skill memory + trust score

    - Call on_trade_close(...) whenever Phase D-1 emits a realized reward event (reduce/close).
    - Stores stats per model_id and optional per symbol.
    """

    def __init__(
        self,
        *,
        state_path: str = "data/memory/skill_memory.json",
        report_path: str = "data/reports/skill_report.json",
        autosave_sec: float = 3.0,
    ) -> None:
        self.state_path = Path(state_path)
        self.report_path = Path(report_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.autosave_sec = float(autosave_sec)
        self._last_save = 0.0

        # structure:
        # { "models": { model_id: SkillStats }, "symbols": { model_id: { sym: SkillStats } } }
        self.models: Dict[str, SkillStats] = {}
        self.symbols: Dict[str, Dict[str, SkillStats]] = {}

        self._load()

    # ---------------------------
    # Public API
    # ---------------------------

    def on_trade_close(
        self,
        *,
        model_id: str,
        symbol: str,
        pnl_net: float,
        confidence: float,
        ts: Optional[float] = None,
    ) -> None:
        """
        Update skill memory from a realized trade outcome.
        pnl_net: realized net PnL in dollars (can be + / - / 0)
        confidence: 0..1 confidence at decision time
        """
        ts = float(ts if ts is not None else time.time())
        symbol = (symbol or "").upper().strip()
        model_id = (model_id or "unknown").strip()

        # update model aggregate
        ms = self.models.get(model_id) or SkillStats()
        self._update_stats(ms, pnl_net=pnl_net, confidence=confidence, ts=ts)
        self.models[model_id] = ms

        # update per-symbol (optional but helpful)
        if model_id not in self.symbols:
            self.symbols[model_id] = {}
        ss = self.symbols[model_id].get(symbol) or SkillStats()
        self._update_stats(ss, pnl_net=pnl_net, confidence=confidence, ts=ts)
        self.symbols[model_id][symbol] = ss

        self._autosave()

    def trust_score(self, model_id: str) -> float:
        """Return 0..1 trust score based on model aggregate stats."""
        ms = self.models.get(model_id)
        if not ms or ms.trades < 3:
            return 0.20  # cautious default until evidence accumulates

        win_rate = ms.wins / max(ms.trades, 1)
        pf = self._profit_factor(ms)
        dd = max(ms.max_drawdown, 0.0)

        # sharpe-ish from trade returns (pnl per trade normalized by stddev)
        sharpe_like = self._sharpe_like(ms)

        # calibration: lower brier is better
        brier = self._brier_mean(ms)

        # Convert components to 0..1 scores (smooth, bounded)
        s_wr = self._sigmoid((win_rate - 0.50) * 6.0)          # 0.5 -> 0.5, 0.6 -> ~0.65
        s_pf = self._sigmoid((min(pf, 3.0) - 1.0) * 2.0)       # PF 1 -> 0.5, PF 2 -> ~0.88
        s_sh = self._sigmoid(sharpe_like * 1.5)                # 0 -> 0.5, 1 -> ~0.82
        s_dd = 1.0 - self._sigmoid((dd - 200.0) / 200.0)       # dd small -> high score; tune threshold later
        s_cal = 1.0 - self._clip01((brier - 0.20) / 0.30)      # brier 0.20 good, 0.50 bad

        # Weighted trust (stable and conservative)
        trust = (
            0.25 * s_wr +
            0.25 * s_pf +
            0.20 * s_sh +
            0.20 * s_dd +
            0.10 * s_cal
        )

        # evidence scaling: trust grows with sample size but caps
        evidence = 1.0 - math.exp(-ms.trades / 25.0)
        trust = 0.15 + evidence * trust  # keep a floor but require evidence

        return self._clip01(trust)

    def write_report(self) -> None:
        """Write a human-readable report JSON for dashboards or inspection."""
        payload = {
            "ts": time.time(),
            "models": {
                mid: {
                    **asdict(stats),
                    "profit_factor": self._profit_factor(stats),
                    "win_rate": (stats.wins / max(stats.trades, 1)),
                    "sharpe_like": self._sharpe_like(stats),
                    "brier": self._brier_mean(stats),
                    "trust": self.trust_score(mid),
                }
                for mid, stats in self.models.items()
            },
        }
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------
    # Internal math + update
    # ---------------------------

    def _update_stats(self, s: SkillStats, *, pnl_net: float, confidence: float, ts: float) -> None:
        pnl_net = float(pnl_net)
        conf = self._clip01(float(confidence))

        s.trades += 1
        if pnl_net > 0:
            s.wins += 1
            s.gross_profit += pnl_net
        elif pnl_net < 0:
            s.losses += 1
            s.gross_loss += pnl_net  # negative

        s.pnl_net += pnl_net

        # realized equity curve + drawdown
        s.equity += pnl_net
        s.peak_equity = max(s.peak_equity, s.equity)
        dd = s.peak_equity - s.equity
        s.max_drawdown = max(s.max_drawdown, dd)

        # sharpe-like via per-trade returns (pnl per trade)
        # Welford update
        r = pnl_net
        delta = r - s.mean_r
        s.mean_r += delta / s.trades
        delta2 = r - s.mean_r
        s.m2_r += delta * delta2

        # Confidence calibration (Brier): (p - outcome)^2
        outcome = 1.0 if pnl_net > 0 else 0.0
        b = (conf - outcome) ** 2
        s.brier_n += 1
        b_delta = b - s.brier_mean
        s.brier_mean += b_delta / s.brier_n
        b_delta2 = b - s.brier_mean
        s.brier_m2 += b_delta * b_delta2

        s.last_ts = ts

    def _profit_factor(self, s: SkillStats) -> float:
        gp = max(s.gross_profit, 0.0)
        gl = abs(min(s.gross_loss, 0.0))
        if gl <= 1e-9:
            return 3.0 if gp > 0 else 1.0
        return gp / gl

    def _sharpe_like(self, s: SkillStats) -> float:
        if s.trades < 5:
            return 0.0
        var = s.m2_r / max(s.trades - 1, 1)
        sd = math.sqrt(max(var, 1e-12))
        return float(s.mean_r / sd)

    def _brier_mean(self, s: SkillStats) -> float:
        if s.brier_n <= 0:
            return 0.33
        return float(s.brier_mean)

    def _autosave(self) -> None:
        now = time.time()
        if now - self._last_save >= self.autosave_sec:
            self._save()
            self.write_report()
            self._last_save = now

    def _save(self) -> None:
        payload = {
            "ts": time.time(),
            "models": {mid: asdict(st) for mid, st in self.models.items()},
            "symbols": {
                mid: {sym: asdict(st) for sym, st in sym_map.items()}
                for mid, sym_map in self.symbols.items()
            },
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.models = {
                mid: SkillStats(**st) for mid, st in (data.get("models") or {}).items()
            }
            self.symbols = {}
            sym_root = data.get("symbols") or {}
            for mid, sym_map in sym_root.items():
                self.symbols[mid] = {sym: SkillStats(**st) for sym, st in sym_map.items()}
        except Exception:
            # corrupt file: start fresh
            self.models = {}
            self.symbols = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        # numerically stable-ish sigmoid
        x = float(x)
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))
