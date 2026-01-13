# runner/phaseG1_backtest_capital.py
# Phase-G.1 — Deterministic Replay with Capital Model

import os
import json
import logging
from pathlib import Path

from ai.feeds.replay_feed import ReplayFeed
from ai.execution.replay_execution_adapter import ReplayExecutionAdapter
from runner.phase26_realtime_live import Phase26RealtimeUltra  # alias provided in Phase-G.1.1

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PhaseG1Backtest")


class PhaseG1Backtest:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg.get("out_dir", "data/backtests/phaseG1_run1"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Force PAPER invariants
        os.environ["MODE"] = "PAPER"
        os.environ["ENV"] = "PAPER_TRADING"

        self.engine = Phase26RealtimeUltra()
        self.engine._disable_live_execution = True
        self.engine._replay_mode = True

        # Inject virtual broker
        self.exec_adapter = ReplayExecutionAdapter(
            starting_equity=cfg.get("starting_equity", 100_000.0)
        )
        self.engine._replay_execution_adapter = self.exec_adapter

        self.trades = []
        self.equity_curve = []

    def run(self, feed: ReplayFeed):
        tick = 0
        for bars in feed.iter_bars():
            tick += 1

            # Provide equity snapshot to engine
            self.engine._external_portfolio_snapshot = self.exec_adapter.snapshot()

            # Drive replay bars
            self.engine._on_replay_bars(bars)

            # Collect fills emitted by Phase-26
            fills = getattr(self.engine, "_replay_fills", [])
            for f in fills:
                self.trades.append(f)
                self.exec_adapter.place_order(
                    symbol=f["symbol"],
                    side=f["side"],
                    qty=f["qty"],
                    price=f["price"],
                )

            # Record equity
            snap = self.exec_adapter.snapshot()
            self.equity_curve.append(
                {"tick": tick, "equity": snap["equity"], "cash": snap["cash"]}
            )

            if tick % 200 == 0:
                log.info("Replay tick %d | equity=%.2f", tick, snap["equity"])

        self._write_outputs()

    def _write_outputs(self):
        (self.out_dir / "trades.json").write_text(
            json.dumps(self.trades, indent=2)
        )
        (self.out_dir / "equity_curve.json").write_text(
            json.dumps(self.equity_curve, indent=2)
        )

        summary = {
            "starting_equity": self.cfg.get("starting_equity", 100_000.0),
            "ending_equity": self.equity_curve[-1]["equity"]
            if self.equity_curve
            else None,
            "num_trades": len(self.trades),
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        log.info("Phase-G.1 complete → %s", self.out_dir)


if __name__ == "__main__":
    cfg = {
        "starting_equity": 100_000.0,
        "out_dir": "data/backtests/phaseG1_run1",
        "ohlcv_csv_by_symbol": {
            "AAPL": "data/ohlcv/AAPL.csv",
            "MSFT": "data/ohlcv/MSFT.csv",
            "TSLA": "data/ohlcv/TSLA.csv",
        },
    }

    feed = ReplayFeed(cfg["ohlcv_csv_by_symbol"])
    PhaseG1Backtest(cfg).run(feed)
