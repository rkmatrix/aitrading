# ai/replay/live_buffer_writer.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .replay_buffer import ReplayBuffer, Transition


class LiveReplayWriter:
    """
    Bridges the live reward loop with a replay buffer and an on-disk JSONL log.

    Each call to `on_row(row)` converts the CSV/summary row into a Transition:
        - ts, symbol, price, position, reward, equity
        - extras: everything else (components, pnl, slippage, etc.)
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        dump_path: Optional[Path] = None,
        flush_every: int = 100,
    ):
        self.buffer = buffer
        self.dump_path = dump_path
        self.flush_every = max(1, int(flush_every))
        self._count = 0

        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't truncate by default; append mode will build history.

    def on_row(self, row: Dict[str, Any]) -> None:
        # Construct Transition from the row emitted by LiveRewardFeedbackLoop
        extras = dict(row)
        ts = float(extras.pop("ts", 0.0))
        symbol = extras.pop("symbol", "")
        price = float(extras.pop("price", 0.0))
        position = float(extras.pop("position", 0.0))
        reward = float(extras.pop("reward", 0.0))
        equity = float(extras.pop("equity", 0.0))

        t = Transition(
            ts=ts,
            symbol=symbol,
            price=price,
            position=position,
            reward=reward,
            equity=equity,
            extras=extras,
        )
        self.buffer.add(t)
        self._count += 1

        if self.dump_path is not None:
            with self.dump_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(t)) + "\n")
