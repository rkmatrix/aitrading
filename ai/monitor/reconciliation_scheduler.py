"""
Position Reconciliation Scheduler
---------------------------------
Schedules and manages automatic position reconciliation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationSchedule:
    """Schedule configuration for reconciliation."""
    interval_seconds: float = 300.0  # Run every 5 minutes
    enabled: bool = True
    max_drift_threshold_pct: float = 1.0  # Alert if drift > 1%
    auto_heal: bool = False  # Automatically heal small discrepancies
    heal_threshold_pct: float = 0.1  # Auto-heal if drift < 0.1%


class ReconciliationScheduler:
    """
    Schedules automatic position reconciliation.
    
    Runs reconciliation at regular intervals and handles
    automatic healing of small discrepancies.
    """
    
    def __init__(
        self,
        reconciler: Any,
        schedule: Optional[ReconciliationSchedule] = None,
        on_reconciliation: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize scheduler.
        
        Args:
            reconciler: PositionReconciler instance
            schedule: Schedule configuration
            on_reconciliation: Callback function called after each reconciliation
        """
        self.reconciler = reconciler
        self.schedule = schedule or ReconciliationSchedule()
        self.on_reconciliation = on_reconciliation
        self.logger = logging.getLogger("ReconciliationScheduler")
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        self._error_count = 0
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            self.logger.warning("Scheduler already running")
            return
        
        if not self.schedule.enabled:
            self.logger.info("Reconciliation scheduler is disabled")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info(
            "Reconciliation scheduler started (interval: %.1fs)",
            self.schedule.interval_seconds
        )
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.logger.info("Reconciliation scheduler stopped")
    
    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Run reconciliation
                self._run_reconciliation()
                
                # Wait for next interval
                time.sleep(self.schedule.interval_seconds)
                
            except Exception as e:
                self._error_count += 1
                self.logger.error("Error in reconciliation scheduler: %s", e, exc_info=True)
                # Continue running even on error
                time.sleep(self.schedule.interval_seconds)
    
    def _run_reconciliation(self) -> None:
        """Run a single reconciliation pass."""
        try:
            self.logger.debug("Running scheduled reconciliation")
            
            # Run reconciler
            result = self.reconciler.run_once()
            
            self._last_run = datetime.now()
            self._run_count += 1
            
            # Check for drift
            summary = result.summary if hasattr(result, "summary") else {}
            severity_pct = summary.get("severity_pct", 0.0)
            
            # Auto-heal if enabled and drift is small
            if self.schedule.auto_heal and severity_pct < self.schedule.heal_threshold_pct:
                self.logger.info(
                    "Auto-healing small drift: %.2f%% < %.2f%%",
                    severity_pct,
                    self.schedule.heal_threshold_pct
                )
                # Trigger healing if reconciler supports it
                if hasattr(self.reconciler, "heal"):
                    try:
                        self.reconciler.heal()
                    except Exception as e:
                        self.logger.warning("Auto-heal failed: %s", e)
            
            # Alert on significant drift
            if severity_pct > self.schedule.max_drift_threshold_pct:
                self.logger.warning(
                    "⚠️ Significant position drift detected: %.2f%% (threshold: %.2f%%)",
                    severity_pct,
                    self.schedule.max_drift_threshold_pct
                )
            
            # Call callback if provided
            if self.on_reconciliation:
                try:
                    self.on_reconciliation({
                        "result": result,
                        "summary": summary,
                        "severity_pct": severity_pct,
                        "run_count": self._run_count,
                    })
                except Exception as e:
                    self.logger.warning("Reconciliation callback failed: %s", e)
            
            self.logger.debug(
                "Reconciliation complete (run #%d, severity: %.2f%%)",
                self._run_count,
                severity_pct
            )
            
        except Exception as e:
            self._error_count += 1
            self.logger.error("Reconciliation run failed: %s", e, exc_info=True)
    
    def run_now(self) -> Optional[Dict[str, Any]]:
        """
        Manually trigger reconciliation immediately.
        
        Returns:
            Reconciliation result dict or None if failed
        """
        try:
            result = self.reconciler.run_once()
            self._last_run = datetime.now()
            self._run_count += 1
            
            summary = result.summary if hasattr(result, "summary") else {}
            
            return {
                "result": result,
                "summary": summary,
                "run_count": self._run_count,
            }
        except Exception as e:
            self.logger.error("Manual reconciliation failed: %s", e, exc_info=True)
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "enabled": self.schedule.enabled,
            "interval_seconds": self.schedule.interval_seconds,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "run_count": self._run_count,
            "error_count": self._error_count,
        }
