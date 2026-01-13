"""
Execution Quality Tracker
Track and optimize execution: slippage, fill quality, market impact
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionQualityConfig:
    """Configuration for execution quality tracking."""
    # State file
    state_file: str = "data/meta/execution_quality.json"
    
    # Slippage thresholds
    max_acceptable_slippage_bps: float = 5.0  # 5 basis points
    
    # Fill quality thresholds
    min_fill_rate: float = 0.95  # 95% fill rate
    max_time_to_fill_sec: float = 5.0  # 5 seconds


@dataclass
class ExecutionRecord:
    """Record of a single execution."""
    symbol: str
    side: str  # "BUY" or "SELL"
    order_qty: float
    expected_price: float
    actual_price: Optional[float] = None
    fill_qty: Optional[float] = None
    fill_time: Optional[datetime] = None
    order_time: datetime = field(default_factory=datetime.now)
    slippage_bps: Optional[float] = None
    market_impact_bps: Optional[float] = None
    order_type: str = "MARKET"
    volatility_regime: Optional[str] = None
    time_of_day_hour: int = field(default_factory=lambda: datetime.now().hour)


class ExecutionQualityTracker:
    """
    Track execution quality metrics.
    
    Features:
    - Slippage tracking
    - Fill quality metrics
    - Time-of-day analysis
    - Volatility regime analysis
    - Order size analysis
    """
    
    def __init__(self, config: Optional[ExecutionQualityConfig] = None):
        """
        Initialize execution quality tracker.
        
        Args:
            config: ExecutionQualityConfig instance (optional)
        """
        self.config = config or ExecutionQualityConfig()
        
        # Execution records
        self.executions: deque = deque(maxlen=1000)
        
        # Aggregated metrics
        self.metrics_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.metrics_by_time: Dict[int, Dict[str, Any]] = {}  # Hour -> metrics
        self.metrics_by_regime: Dict[str, Dict[str, Any]] = {}
        
        # Load persisted state
        self._load_state()
        
        logger.info("ExecutionQualityTracker initialized")
    
    def record_order(
        self,
        symbol: str,
        side: str,
        order_qty: float,
        expected_price: float,
        order_type: str = "MARKET",
        volatility_regime: Optional[str] = None,
    ) -> ExecutionRecord:
        """
        Record an order submission.
        
        Args:
            symbol: Symbol
            side: "BUY" or "SELL"
            order_qty: Order quantity
            expected_price: Expected fill price
            order_type: Order type
            volatility_regime: Volatility regime (optional)
        
        Returns:
            ExecutionRecord for tracking
        """
        record = ExecutionRecord(
            symbol=symbol.upper(),
            side=side.upper(),
            order_qty=order_qty,
            expected_price=expected_price,
            order_type=order_type,
            volatility_regime=volatility_regime,
            time_of_day_hour=datetime.now().hour,
        )
        
        return record
    
    def record_fill(
        self,
        record: ExecutionRecord,
        actual_price: float,
        fill_qty: Optional[float] = None,
        fill_time: Optional[datetime] = None,
    ) -> None:
        """
        Record order fill.
        
        Args:
            record: ExecutionRecord from record_order
            actual_price: Actual fill price
            fill_qty: Filled quantity (defaults to order_qty)
            fill_time: Fill timestamp (defaults to now)
        """
        record.actual_price = actual_price
        record.fill_qty = fill_qty or record.order_qty
        record.fill_time = fill_time or datetime.now()
        
        # Calculate slippage
        if record.expected_price > 0:
            price_diff = actual_price - record.expected_price
            if record.side == "SELL":
                price_diff = -price_diff  # For sells, negative slippage is bad
            
            record.slippage_bps = (price_diff / record.expected_price) * 10000
        
        # Calculate fill rate
        fill_rate = record.fill_qty / record.order_qty if record.order_qty > 0 else 0.0
        
        # Calculate time to fill
        time_to_fill = (record.fill_time - record.order_time).total_seconds()
        
        # Store record
        self.executions.append(record)
        
        # Update aggregated metrics
        self._update_metrics(record)
        
        # Save state periodically
        if len(self.executions) % 10 == 0:
            self._save_state()
    
    def _update_metrics(self, record: ExecutionRecord) -> None:
        """Update aggregated metrics."""
        # By symbol
        symbol = record.symbol
        if symbol not in self.metrics_by_symbol:
            self.metrics_by_symbol[symbol] = {
                "total_orders": 0,
                "total_fills": 0,
                "total_slippage_bps": 0.0,
                "avg_slippage_bps": 0.0,
                "fill_rate": 0.0,
                "avg_time_to_fill": 0.0,
            }
        
        metrics = self.metrics_by_symbol[symbol]
        metrics["total_orders"] += 1
        
        if record.actual_price is not None:
            metrics["total_fills"] += 1
            if record.slippage_bps is not None:
                metrics["total_slippage_bps"] += record.slippage_bps
                metrics["avg_slippage_bps"] = metrics["total_slippage_bps"] / metrics["total_fills"]
            
            fill_rate = record.fill_qty / record.order_qty if record.order_qty > 0 else 0.0
            metrics["fill_rate"] = (metrics["fill_rate"] * (metrics["total_fills"] - 1) + fill_rate) / metrics["total_fills"]
            
            time_to_fill = (record.fill_time - record.order_time).total_seconds()
            metrics["avg_time_to_fill"] = (metrics["avg_time_to_fill"] * (metrics["total_fills"] - 1) + time_to_fill) / metrics["total_fills"]
        
        # By time of day
        hour = record.time_of_day_hour
        if hour not in self.metrics_by_time:
            self.metrics_by_time[hour] = {
                "total_orders": 0,
                "avg_slippage_bps": 0.0,
                "total_slippage_bps": 0.0,
                "fill_count": 0,
            }
        
        time_metrics = self.metrics_by_time[hour]
        time_metrics["total_orders"] += 1
        
        if record.slippage_bps is not None:
            time_metrics["fill_count"] += 1
            time_metrics["total_slippage_bps"] += record.slippage_bps
            time_metrics["avg_slippage_bps"] = time_metrics["total_slippage_bps"] / time_metrics["fill_count"]
        
        # By volatility regime
        if record.volatility_regime:
            regime = record.volatility_regime
            if regime not in self.metrics_by_regime:
                self.metrics_by_regime[regime] = {
                    "total_orders": 0,
                    "avg_slippage_bps": 0.0,
                    "total_slippage_bps": 0.0,
                    "fill_count": 0,
                }
            
            regime_metrics = self.metrics_by_regime[regime]
            regime_metrics["total_orders"] += 1
            
            if record.slippage_bps is not None:
                regime_metrics["fill_count"] += 1
                regime_metrics["total_slippage_bps"] += record.slippage_bps
                regime_metrics["avg_slippage_bps"] = regime_metrics["total_slippage_bps"] / regime_metrics["fill_count"]
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall execution quality summary."""
        if len(self.executions) == 0:
            return {
                "total_orders": 0,
                "avg_slippage_bps": 0.0,
                "fill_rate": 0.0,
                "avg_time_to_fill": 0.0,
            }
        
        filled = [e for e in self.executions if e.actual_price is not None]
        
        if len(filled) == 0:
            return {
                "total_orders": len(self.executions),
                "avg_slippage_bps": 0.0,
                "fill_rate": 0.0,
                "avg_time_to_fill": 0.0,
            }
        
        avg_slippage = np.mean([e.slippage_bps for e in filled if e.slippage_bps is not None])
        fill_rate = np.mean([e.fill_qty / e.order_qty for e in filled if e.order_qty > 0])
        avg_time_to_fill = np.mean([
            (e.fill_time - e.order_time).total_seconds()
            for e in filled if e.fill_time and e.order_time
        ])
        
        return {
            "total_orders": len(self.executions),
            "filled_orders": len(filled),
            "avg_slippage_bps": float(avg_slippage) if avg_slippage else 0.0,
            "fill_rate": float(fill_rate) if fill_rate else 0.0,
            "avg_time_to_fill": float(avg_time_to_fill) if avg_time_to_fill else 0.0,
            "by_symbol": self.metrics_by_symbol,
            "by_time": self.metrics_by_time,
            "by_regime": self.metrics_by_regime,
        }
    
    def get_best_execution_times(self) -> List[Dict[str, Any]]:
        """Get best execution times based on historical data."""
        if not self.metrics_by_time:
            return []
        
        # Sort by average slippage (lower is better)
        sorted_times = sorted(
            self.metrics_by_time.items(),
            key=lambda x: x[1].get("avg_slippage_bps", 999) if x[1].get("fill_count", 0) > 0 else 999
        )
        
        return [
            {
                "hour": hour,
                "avg_slippage_bps": metrics.get("avg_slippage_bps", 0.0),
                "total_orders": metrics.get("total_orders", 0),
            }
            for hour, metrics in sorted_times[:5]  # Top 5
        ]
    
    def _save_state(self) -> None:
        """Save execution quality state to file."""
        try:
            state_file = Path(self.config.state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                "executions": [
                    {
                        "symbol": e.symbol,
                        "side": e.side,
                        "order_qty": e.order_qty,
                        "expected_price": e.expected_price,
                        "actual_price": e.actual_price,
                        "fill_qty": e.fill_qty,
                        "fill_time": e.fill_time.isoformat() if e.fill_time else None,
                        "order_time": e.order_time.isoformat(),
                        "slippage_bps": e.slippage_bps,
                        "order_type": e.order_type,
                        "volatility_regime": e.volatility_regime,
                        "time_of_day_hour": e.time_of_day_hour,
                    }
                    for e in list(self.executions)[-100]  # Keep last 100
                ],
                "metrics_by_symbol": self.metrics_by_symbol,
                "metrics_by_time": {str(k): v for k, v in self.metrics_by_time.items()},
                "metrics_by_regime": self.metrics_by_regime,
            }
            
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save execution quality state: %s", e)
    
    def _load_state(self) -> None:
        """Load execution quality state from file."""
        try:
            state_file = Path(self.config.state_file)
            if not state_file.exists():
                return
            
            with state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            # Load executions
            for e_data in state_data.get("executions", []):
                record = ExecutionRecord(
                    symbol=e_data["symbol"],
                    side=e_data["side"],
                    order_qty=float(e_data["order_qty"]),
                    expected_price=float(e_data["expected_price"]),
                    actual_price=float(e_data["actual_price"]) if e_data.get("actual_price") else None,
                    fill_qty=float(e_data["fill_qty"]) if e_data.get("fill_qty") else None,
                    fill_time=datetime.fromisoformat(e_data["fill_time"]) if e_data.get("fill_time") else None,
                    order_time=datetime.fromisoformat(e_data["order_time"]),
                    slippage_bps=float(e_data["slippage_bps"]) if e_data.get("slippage_bps") else None,
                    order_type=e_data.get("order_type", "MARKET"),
                    volatility_regime=e_data.get("volatility_regime"),
                    time_of_day_hour=int(e_data.get("time_of_day_hour", 0)),
                )
                self.executions.append(record)
            
            # Load metrics
            self.metrics_by_symbol = state_data.get("metrics_by_symbol", {})
            self.metrics_by_time = {
                int(k): v for k, v in state_data.get("metrics_by_time", {}).items()
            }
            self.metrics_by_regime = state_data.get("metrics_by_regime", {})
        except Exception as e:
            logger.warning("Failed to load execution quality state: %s", e)
