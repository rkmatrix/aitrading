# ai/guardians/market_hours_guardian.py
"""
Market-Hours Guardian (Phase 53.10)
-----------------------------------
Handles regular and extended trading-session detection with auto-detect
from Alpaca's /v2/clock endpoint.  Also prevents property shadowing
errors ('bool' object is not callable) permanently.
"""

from __future__ import annotations
import os, json, datetime, urllib.request, logging
from typing import Optional, Literal
from tools.telegram_alerts import notify

try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except Exception:
    _NY_TZ = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

Mode = Literal["regular", "extended", "auto"]


class MarketHoursGuardian:
    def __init__(self,
                 base_url: Optional[str] = None,
                 mode: Mode = "auto"):
        self.base_url = base_url or os.getenv(
            "APCA_API_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self.key = os.getenv("APCA_API_KEY_ID") or ""
        self.secret = os.getenv("APCA_API_SECRET_KEY") or ""
        self.mode: Mode = mode

        # internal trackers (prefix with underscore so they can’t collide)
        self._last_allowed: Optional[bool] = None
        self._last_open_flag: Optional[bool] = None
        self._last_is_ext_window: Optional[bool] = None
        self._last_ts: Optional[str] = None
        self._last_clock: Optional[dict] = None

    # ------------------------------------------------------------------
    def _now_et(self) -> datetime.datetime:
        now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        if _NY_TZ:
            return now_utc.astimezone(_NY_TZ)
        return (now_utc - datetime.timedelta(hours=5)).replace(tzinfo=None)

    # ------------------------------------------------------------------
    def _fetch_clock(self) -> Optional[dict]:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v2/clock",
                headers={
                    "APCA-API-KEY-ID": self.key,
                    "APCA-API-SECRET-KEY": self.secret,
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self._last_clock = data
                    return data
        except Exception as e:
            logger.debug("Clock fetch failed: %s", e)
        return None

    # ------------------------------------------------------------------
    @property
    def extended_session_active(self) -> bool:
        """
        True if current ET time is in pre-market (7–9:30 AM ET)
        or after-hours (4–8 PM ET) window.
        """
        now_et = self._now_et()
        tmins = now_et.hour * 60 + now_et.minute
        return (7 * 60 <= tmins < 9 * 60 + 30) or (16 * 60 <= tmins <= 20 * 60)

    @property
    def regular_session_active(self) -> bool:
        """True if current ET time is within 9:30–16:00 ET."""
        now_et = self._now_et()
        tmins = now_et.hour * 60 + now_et.minute
        return (9 * 60 + 30) <= tmins < (16 * 60)

    # ------------------------------------------------------------------
    def allowed_now(self) -> bool:
        """
        Returns True if trading is allowed per current mode.
        Sends Telegram alerts on OPEN/CLOSE transitions.
        """
        clock = self._fetch_clock()
        is_open_flag = bool(clock.get("is_open")) if isinstance(clock, dict) and "is_open" in clock else None

        in_regular = self.regular_session_active
        in_extended = self.extended_session_active

        # ---- determine allowance ----
        if self.mode == "regular":
            allowed = is_open_flag if is_open_flag is not None else in_regular
        elif self.mode == "extended":
            allowed = in_regular or in_extended
        else:  # auto
            if is_open_flag is True:
                allowed = True
            elif in_regular:
                allowed = True
            else:
                allowed = in_extended

        # ---- transition alerts ----
        if self._last_allowed is not None and allowed != self._last_allowed:
            if allowed:
                msg = "🟢 Market opened – resuming trading."
                if self.mode in ("auto", "extended") and in_extended and not in_regular:
                    msg += " (Extended session)"
                logger.info(msg)
                notify(msg, kind="system")
            else:
                msg = "🔴 Market closed – pausing trading."
                logger.info(msg)
                notify(msg, kind="system")

        # store for diagnostics
        self._last_allowed = allowed
        self._last_open_flag = is_open_flag
        self._last_is_ext_window = in_extended
        self._last_ts = (clock or {}).get("timestamp") if clock else self._now_et().isoformat()

        return allowed

    # ------------------------------------------------------------------
    def require_open(self) -> bool:
        ok = self.allowed_now()
        if not ok:
            logger.info("⏸️ Trading paused (market not allowed in mode=%s).", self.mode)
        return ok
