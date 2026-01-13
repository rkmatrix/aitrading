"""
Structured Logger
----------------
Provides structured logging for critical events with context.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logger for critical trading events.
    
    Logs events in JSON format with consistent structure:
    {
        "timestamp": "ISO8601",
        "level": "INFO|WARNING|ERROR|CRITICAL",
        "event_type": "order|risk|error|system",
        "message": "Human readable message",
        "context": {...}
    }
    """
    
    def __init__(
        self,
        name: str = "StructuredLogger",
        log_file: Optional[Path] = None,
        console: bool = True,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.console = console
        
        # Setup file handler if log file specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
        
        # Setup console handler if enabled
        if self.console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.setLevel(logging.INFO)
    
    def _log_structured(
        self,
        level: str,
        event_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None,
    ) -> None:
        """Internal method to log structured event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "event_type": event_type,
            "message": message,
            "context": context or {},
        }
        
        if exc_info:
            log_entry["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exc(),
            }
        
        # Log as JSON string
        json_str = json.dumps(log_entry, default=str)
        
        # Use appropriate logging level
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json_str)
    
    def log_order(
        self,
        message: str,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log order event."""
        context = {
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "price": float(price) if price is not None else None,
            "order_id": order_id,
            "status": status,
            **kwargs,
        }
        self._log_structured("INFO", "order", message, context)
    
    def log_risk(
        self,
        message: str,
        symbol: Optional[str] = None,
        risk_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log risk event."""
        context = {
            "symbol": symbol,
            "risk_type": risk_type,
            **kwargs,
        }
        self._log_structured("WARNING", "risk", message, context)
    
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        """Log error event."""
        context = kwargs.copy()
        self._log_structured("ERROR", "error", message, context, exc_info=error)
    
    def log_system(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log system event."""
        context = {
            "component": component,
            **kwargs,
        }
        self._log_structured("INFO", "system", message, context)
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log metric."""
        context = {
            "metric_name": metric_name,
            "value": float(value),
            "unit": unit,
            **kwargs,
        }
        self._log_structured("INFO", "metric", f"Metric: {metric_name}", context)
