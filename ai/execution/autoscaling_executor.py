from __future__ import annotations
import logging, time
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import yaml  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AutoScaleConfig:
    base_interval_sec: float = 3.0
    min_interval_sec: float = 0.5
    max_interval_sec: float = 10.0
    low_vol_threshold: float = 0.5
    high_vol_threshold: float = 1.5
    max_consecutive_fast_ticks: int = 50
    cooldown_sec: float = 5.0
    log_interval_sec: float = 30.0


class AdaptiveAutoScalingExecutor:
    """
    Wraps a generic 'tick' function with an adaptive sleep interval.

    - If volatility increases → interval shrinks (more frequent ticks).
    - If volatility decreases → interval grows (less frequent ticks).
    - Includes safeguards to avoid spinning too fast for too long.

    Expected callbacks:

        tick_fn(ctx) -> None
        state_fn() -> {"volatility": float, ...}
    """

    def __init__(self, cfg: AutoScaleConfig) -> None:
        self.cfg = cfg
        self._tick_idx: int = 0
        self._fast_tick_count: int = 0
        self._last_log_ts: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "AdaptiveAutoScalingExecutor":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        asc = raw.get("autoscale", {}) or {}
        cfg = AutoScaleConfig(
            base_interval_sec=float(asc.get("base_interval_sec", 3.0)),
            min_interval_sec=float(asc.get("min_interval_sec", 0.5)),
            max_interval_sec=float(asc.get("max_interval_sec", 10.0)),
            low_vol_threshold=float(asc.get("low_vol_threshold", 0.5)),
            high_vol_threshold=float(asc.get("high_vol_threshold", 1.5)),
            max_consecutive_fast_ticks=int(asc.get("max_consecutive_fast_ticks", 50)),
            cooldown_sec=float(asc.get("cooldown_sec", 5.0)),
            log_interval_sec=float(raw.get("logging", {}).get("log_interval_sec", 30.0)),
        )
        return cls(cfg)

    # ------------------------------------------------------------------ #

    def _compute_interval(self, volatility: Optional[float]) -> float:
        if volatility is None:
            return self.cfg.base_interval_sec

        interval = self.cfg.base_interval_sec

        if volatility <= self.cfg.low_vol_threshold:
            interval *= 1.5
        elif volatility >= self.cfg.high_vol_threshold:
            interval *= 0.5

        interval = max(self.cfg.min_interval_sec, min(self.cfg.max_interval_sec, interval))
        return interval

    def run(
        self,
        tick_fn: Callable[[Dict[str, Any]], None],
        state_fn: Callable[[], Dict[str, Any]],
        *,
        running_flag_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        logger.info("🚀 AdaptiveAutoScalingExecutor starting…")
        while True:
            if running_flag_fn and not running_flag_fn():
                logger.info("Stopping AdaptiveAutoScalingExecutor – running_flag_fn=False")
                break

            self._tick_idx += 1
            loop_start = time.time()

            state = {}
            try:
                state = state_fn() or {}
            except Exception:
                logger.exception("state_fn exploded; using empty state")
                state = {}

            vol = state.get("volatility")
            interval = self._compute_interval(vol)

            fast_mode = interval <= (self.cfg.base_interval_sec * 0.75)
            if fast_mode:
                self._fast_tick_count += 1
            else:
                self._fast_tick_count = 0

            if self._fast_tick_count > self.cfg.max_consecutive_fast_ticks:
                logger.warning(
                    "Hit max_consecutive_fast_ticks=%s, enforcing cooldown %ss",
                    self.cfg.max_consecutive_fast_ticks,
                    self.cfg.cooldown_sec,
                )
                time.sleep(self.cfg.cooldown_sec)
                self._fast_tick_count = 0

            ctx = {
                "tick_idx": self._tick_idx,
                "volatility": vol,
                "interval_sec": interval,
                "fast_mode": fast_mode,
                "state": state,
            }

            try:
                tick_fn(ctx)
            except Exception:
                logger.exception("tick_fn exploded at tick=%s", self._tick_idx)

            elapsed = time.time() - loop_start
            sleep_for = max(0.0, interval - elapsed)

            now = time.time()
            if now - self._last_log_ts >= self.cfg.log_interval_sec:
                logger.info(
                    "AutoScale tick=%s vol=%s interval=%.3fs fast_ticks=%s",
                    self._tick_idx,
                    None if vol is None else f"{vol:.3f}",
                    interval,
                    self._fast_tick_count,
                )
                self._last_log_ts = now

            if sleep_for > 0:
                time.sleep(sleep_for)
