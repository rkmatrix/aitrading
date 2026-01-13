"""
Kill Switch Monitor with Heartbeat
----------------------------------
Monitors kill switch state and provides heartbeat mechanism to verify
it's being checked regularly.
"""

from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch monitoring."""
    flag_path: str = "data/runtime/trading_disabled.flag"
    heartbeat_interval_sec: float = 30.0  # How often to check kill switch
    heartbeat_timeout_sec: float = 120.0  # Alert if no heartbeat for this long
    auto_activate_on_critical_error: bool = True


class KillSwitchMonitor:
    """
    Monitors kill switch state and provides heartbeat mechanism.
    
    Features:
    - Tracks kill switch state changes
    - Provides heartbeat mechanism to verify monitoring is active
    - Can auto-activate kill switch on critical errors
    - Logs all state changes
    """
    
    def __init__(self, config: Optional[KillSwitchConfig] = None):
        self.config = config or KillSwitchConfig()
        self.flag_path = Path(self.config.flag_path)
        self._last_state: Optional[bool] = None
        self._last_check_time: Optional[datetime] = None
        self._last_heartbeat_time: Optional[datetime] = None
        self._state_change_count = 0
        self.logger = logging.getLogger("KillSwitchMonitor")
        
        # Ensure directory exists
        self.flag_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self._last_state = self.is_active()
        self._last_check_time = datetime.now()
        self._last_heartbeat_time = datetime.now()
    
    def is_active(self) -> bool:
        """
        Check if kill switch is currently active.
        
        Returns:
            True if kill switch is active (trading should stop)
        """
        if not self.flag_path.exists():
            return False
        
        # Try to read JSON format first
        try:
            content = self.flag_path.read_text(encoding="utf-8").strip()
            if content.startswith("{"):
                data = json.loads(content)
                return bool(data.get("kill", data.get("enabled", True)))
        except Exception:
            pass
        
        # Fallback to text file check
        try:
            content = self.flag_path.read_text(encoding="utf-8").strip().upper()
            return content in ("1", "TRUE", "YES", "ON", "KILL", "STOP", "HALT", "TRADING_DISABLED")
        except Exception:
            # If file exists but can't read, assume active (fail-safe)
            return True
    
    def check(self) -> Dict[str, Any]:
        """
        Check kill switch state and update heartbeat.
        
        Returns:
            Dict with state information:
            {
                "active": bool,
                "state_changed": bool,
                "last_state": bool | None,
                "time_since_last_check": float,
                "heartbeat_ok": bool
            }
        """
        now = datetime.now()
        current_state = self.is_active()
        state_changed = False
        
        if self._last_state is not None and current_state != self._last_state:
            state_changed = True
            self._state_change_count += 1
            
            if current_state:
                self.logger.error(
                    "🚨 KILL SWITCH ACTIVATED at %s (path: %s)",
                    now.isoformat(),
                    self.flag_path
                )
            else:
                self.logger.warning(
                    "⚠️ KILL SWITCH DEACTIVATED at %s (path: %s)",
                    now.isoformat(),
                    self.flag_path
                )
        
        self._last_state = current_state
        self._last_check_time = now
        self._last_heartbeat_time = now
        
        time_since_check = (
            (now - self._last_check_time).total_seconds()
            if self._last_check_time
            else 0.0
        )
        
        heartbeat_ok = (
            (now - self._last_heartbeat_time).total_seconds()
            <= self.config.heartbeat_timeout_sec
        )
        
        return {
            "active": current_state,
            "state_changed": state_changed,
            "last_state": self._last_state,
            "time_since_last_check": time_since_check,
            "heartbeat_ok": heartbeat_ok,
            "state_change_count": self._state_change_count,
        }
    
    def activate(self, reason: str = "manual", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Activate kill switch.
        
        Args:
            reason: Reason for activation
            metadata: Optional metadata to include
        """
        try:
            data = {
                "kill": True,
                "enabled": True,
                "status": "halted",
                "reason": reason,
                "activated_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            
            self.flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.flag_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            self.logger.error(
                "🚨 KILL SWITCH ACTIVATED: %s (path: %s)",
                reason,
                self.flag_path
            )
            
            # Update state
            self._last_state = True
            self._last_check_time = datetime.now()
            
        except Exception as e:
            self.logger.error("Failed to activate kill switch: %s", e, exc_info=True)
    
    def deactivate(self, reason: str = "manual") -> None:
        """
        Deactivate kill switch.
        
        Args:
            reason: Reason for deactivation
        """
        try:
            if self.flag_path.exists():
                self.flag_path.unlink()
            
            self.logger.warning(
                "⚠️ KILL SWITCH DEACTIVATED: %s",
                reason
            )
            
            # Update state
            self._last_state = False
            self._last_check_time = datetime.now()
            
        except Exception as e:
            self.logger.error("Failed to deactivate kill switch: %s", e, exc_info=True)
    
    def get_heartbeat_status(self) -> Dict[str, Any]:
        """
        Get heartbeat status to verify monitoring is active.
        
        Returns:
            Dict with heartbeat information
        """
        now = datetime.now()
        
        if self._last_heartbeat_time is None:
            return {
                "heartbeat_ok": False,
                "last_heartbeat": None,
                "time_since_heartbeat": None,
                "status": "no_heartbeat",
            }
        
        time_since = (now - self._last_heartbeat_time).total_seconds()
        heartbeat_ok = time_since <= self.config.heartbeat_timeout_sec
        
        return {
            "heartbeat_ok": heartbeat_ok,
            "last_heartbeat": self._last_heartbeat_time.isoformat(),
            "time_since_heartbeat": time_since,
            "status": "ok" if heartbeat_ok else "stale",
        }
    
    def activate_on_critical_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Automatically activate kill switch on critical error.
        
        Args:
            error: The critical error that occurred
            context: Optional context about the error
        """
        if not self.config.auto_activate_on_critical_error:
            return
        
        reason = f"Critical error: {type(error).__name__}: {str(error)}"
        metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "auto_activated": True,
        }
        
        self.activate(reason=reason, metadata=metadata)
