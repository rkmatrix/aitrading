# ai/reward/live_feedback_loop.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .sources import EventSource, DummySource, Event
from .calculators import RewardCalculator, RewardWeights, RewardParams
from .sinks import CsvSink, SummarySink

log = logging.getLogger("LiveRewardLoop")


@dataclass
class LoopConfig:
    interval_sec: float = 1.0
    bursts: int = 1
    max_events_per_burst: int = 10


class LiveRewardFeedbackLoop:
    def __init__(
        self,
        source: EventSource,
        calc: RewardCalculator,
        csv_sink: CsvSink,
        summary_sink: SummarySink,
        symbols: List[str],
        loop_cfg: LoopConfig,
        replay_writer: Optional["LiveReplayWriter"] = None,
    ):
        self.source = source
        self.calc = calc
        self.csv = csv_sink
        self.summary = summary_sink
        self.symbols = symbols
        self.loop_cfg = loop_cfg
        self.replay_writer = replay_writer

    @classmethod
    def from_config(cls, cfg: Dict) -> "LiveRewardFeedbackLoop":
        symbols = list(cfg.get("symbols", ["AAPL"]))
        loop_cfg = cfg.get("loop", {})
        lc = LoopConfig(
            interval_sec=float(loop_cfg.get("interval_sec", 1.0)),
            bursts=int(loop_cfg.get("bursts", 1)),
            max_events_per_burst=int(loop_cfg.get("max_events_per_burst", 10)),
        )

        # --------------------- Source ---------------------
        src_cfg = cfg.get("source", {})
        src_type = (src_cfg.get("type") or "dummy").lower()

        if src_type == "dummy":
            d = src_cfg.get("dummy", {})
            source = DummySource(
                symbols=symbols,
                seed=int(d.get("seed", 42)),
                start_price=float(d.get("price", {}).get("start", 100.0)),
                drift_perc=float(d.get("price", {}).get("drift_perc", 0.0005)),
                vol_perc=float(d.get("price", {}).get("vol_perc", 0.01)),
                event_rate_per_symbol=int(d.get("event_rate_per_symbol", 2)),
            )
        elif src_type == "alpaca":
            from .alpaca_fills_adapter import AlpacaFillsSource

            a_cfg = src_cfg.get("alpaca", {})
            source = AlpacaFillsSource(symbols=symbols, cfg=a_cfg)
        else:
            raise ValueError(f"Unknown source.type={src_type}")

        # ------------------ Reward calc -------------------
        wcfg = cfg.get("reward", {}).get("weights", {})
        pcfg = cfg.get("reward", {}).get("params", {})
        weights = RewardWeights(
            pnl_realized=float(wcfg.get("pnl_realized", 1.0)),
            pnl_unrealized=float(wcfg.get("pnl_unrealized", 0.0)),
            drawdown_penalty=float(wcfg.get("drawdown_penalty", -0.5)),
            slippage_penalty=float(wcfg.get("slippage_penalty", -0.2)),
            risk_penalty=float(wcfg.get("risk_penalty", -0.3)),
        )
        params = RewardParams(
            max_drawdown_window=int(pcfg.get("max_drawdown_window", 200)),
            risk_target_vol=float(pcfg.get("risk_target_vol", 0.02)),
            slippage_bps=float(pcfg.get("slippage_bps", 2.0)),
        )
        calc = RewardCalculator(weights, params)

        # ---------------------- Sinks ---------------------
        sinks_cfg = cfg.get("sinks", {})
        csv_path = Path(
            sinks_cfg.get("csv_path", "data/rewards/phase54_events.csv")
        )
        summary_path = Path(
            sinks_cfg.get("summary_path", "data/reports/phase54_reward_summary.json")
        )
        notify_cfg = sinks_cfg.get("notify", {})
        csv = CsvSink(csv_path)
        summary = SummarySink(
            summary_path,
            z_alert_threshold=float(notify_cfg.get("anomaly_threshold", 3.0)),
            notify_kind=(cfg.get("telegram", {}) and cfg["telegram"].get("kind"))
            or "guardian",
            only_on_anomaly=bool(notify_cfg.get("only_on_anomaly", True)),
        )

        # ------------------ Replay writer -----------------
        replay_writer = None
        r_cfg = cfg.get("replay", {})
        if r_cfg.get("enabled", False):
            from ai.replay.replay_buffer import ReplayBuffer
            from ai.replay.live_buffer_writer import LiveReplayWriter

            capacity = int(r_cfg.get("max_size", 100_000))
            dump_path_str = r_cfg.get(
                "dump_path", "data/replay/phase55_replay.jsonl"
            )
            flush_every = int(r_cfg.get("flush_every", 100))

            buffer = ReplayBuffer(capacity=capacity)
            dump_path = Path(dump_path_str) if dump_path_str else None
            replay_writer = LiveReplayWriter(
                buffer=buffer, dump_path=dump_path, flush_every=flush_every
            )
            log.info(
                "Replay buffer enabled: capacity=%d, dump_path=%s",
                capacity,
                dump_path,
            )

        return cls(source, calc, csv, summary, symbols, lc, replay_writer)

    def _row_from_event(self, e: Event, out: Dict) -> Dict:
        comps = out["components"]
        return {
            "ts": e.ts,
            "symbol": e.symbol,
            "price": e.px,
            "position": e.position,
            "realized_pnl": e.realized_pnl,
            "unrealized_pnl": e.unrealized_pnl,
            "slippage": e.slippage,
            "risk": e.risk,
            "equity": out["equity"],
            "reward": out["reward"],
            "comp_pnl_realized": comps["pnl_realized"],
            "comp_pnl_unrealized": comps["pnl_unrealized"],
            "comp_drawdown": comps["drawdown"],
            "comp_slippage": comps["slippage"],
            "comp_risk_dev": comps["risk_dev"],
            "trade_shares": e.meta.get("trade_shares", 0),
        }

    def run_burst(self) -> int:
        max_ev = self.loop_cfg.max_events_per_burst
        events = self.source.poll(max_ev)
        processed = 0
        last_row = None
        for e in events:
            out = self.calc.compute(e)
            row = self._row_from_event(e, out)
            self.csv.write(row)
            self.summary.update(row["reward"])
            if self.replay_writer is not None:
                self.replay_writer.on_row(row)
            last_row = row
            processed += 1
        if last_row:
            self.summary.flush(last_row)
        return processed

    def close(self):
        try:
            self.source.close()
        except Exception:
            pass
