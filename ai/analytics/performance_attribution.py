"""
Performance Attribution System
Understand what's working: P&L by signal type, symbol, time, regime
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    side: str  # "LONG" or "SHORT"
    pnl: float
    signal_type: Optional[str] = None
    regime: Optional[str] = None
    duration_seconds: Optional[float] = None


class PerformanceAttribution:
    """
    Performance attribution analyzer.
    
    Features:
    - P&L by signal type
    - P&L by symbol
    - P&L by time of day
    - P&L by market regime
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
    - Trade analysis (best/worst trades, duration)
    """
    
    def __init__(self, state_file: str = "data/meta/performance_attribution.json"):
        """
        Initialize performance attribution.
        
        Args:
            state_file: Path to state file
        """
        self.state_file = state_file
        self.trades: List[TradeRecord] = []
        
        # Load persisted state
        self._load_state()
        
        logger.info("PerformanceAttribution initialized with %d trades", len(self.trades))
    
    def record_trade(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        qty: float,
        side: str,
        signal_type: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            qty: Quantity
            side: "LONG" or "SHORT"
            signal_type: Signal type (optional)
            regime: Market regime (optional)
        """
        # Calculate P&L
        if side.upper() == "LONG":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty
        
        duration = (exit_time - entry_time).total_seconds()
        
        trade = TradeRecord(
            symbol=symbol.upper(),
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            side=side.upper(),
            pnl=pnl,
            signal_type=signal_type,
            regime=regime,
            duration_seconds=duration,
        )
        
        self.trades.append(trade)
        
        # Save state periodically
        if len(self.trades) % 10 == 0:
            self._save_state()
    
    def get_attribution_by_signal_type(self) -> Dict[str, Any]:
        """Get P&L attribution by signal type."""
        if not self.trades:
            return {}
        
        by_signal = defaultdict(lambda: {"trades": [], "total_pnl": 0.0, "count": 0})
        
        for trade in self.trades:
            signal = trade.signal_type or "unknown"
            by_signal[signal]["trades"].append(trade.pnl)
            by_signal[signal]["total_pnl"] += trade.pnl
            by_signal[signal]["count"] += 1
        
        result = {}
        for signal, data in by_signal.items():
            pnls = data["trades"]
            result[signal] = {
                "total_pnl": data["total_pnl"],
                "count": data["count"],
                "avg_pnl": np.mean(pnls) if pnls else 0.0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
                "avg_win": np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0.0,
                "avg_loss": np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0.0,
            }
        
        return result
    
    def get_attribution_by_symbol(self) -> Dict[str, Any]:
        """Get P&L attribution by symbol."""
        if not self.trades:
            return {}
        
        by_symbol = defaultdict(lambda: {"trades": [], "total_pnl": 0.0, "count": 0})
        
        for trade in self.trades:
            symbol = trade.symbol
            by_symbol[symbol]["trades"].append(trade.pnl)
            by_symbol[symbol]["total_pnl"] += trade.pnl
            by_symbol[symbol]["count"] += 1
        
        result = {}
        for symbol, data in by_symbol.items():
            pnls = data["trades"]
            result[symbol] = {
                "total_pnl": data["total_pnl"],
                "count": data["count"],
                "avg_pnl": np.mean(pnls) if pnls else 0.0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
            }
        
        return result
    
    def get_attribution_by_regime(self) -> Dict[str, Any]:
        """Get P&L attribution by market regime."""
        if not self.trades:
            return {}
        
        by_regime = defaultdict(lambda: {"trades": [], "total_pnl": 0.0, "count": 0})
        
        for trade in self.trades:
            regime = trade.regime or "unknown"
            by_regime[regime]["trades"].append(trade.pnl)
            by_regime[regime]["total_pnl"] += trade.pnl
            by_regime[regime]["count"] += 1
        
        result = {}
        for regime, data in by_regime.items():
            pnls = data["trades"]
            result[regime] = {
                "total_pnl": data["total_pnl"],
                "count": data["count"],
                "avg_pnl": np.mean(pnls) if pnls else 0.0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
            }
        
        return result
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))  # Annualized
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
    
    def get_risk_adjusted_metrics(self) -> Dict[str, Any]:
        """Get risk-adjusted performance metrics."""
        if not self.trades:
            return {}
        
        # Calculate daily returns (simplified)
        pnls = [t.pnl for t in self.trades]
        
        # Calculate cumulative returns
        cumulative = np.cumsum(pnls)
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Calculate returns as percentages (simplified)
        returns = [p / 1000.0 for p in pnls]  # Normalize by assumed capital
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        
        total_return = float(np.sum(pnls))
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
        }
    
    def get_best_worst_trades(self, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get best and worst trades."""
        if not self.trades:
            return {"best": [], "worst": []}
        
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)
        
        best = [
            {
                "symbol": t.symbol,
                "pnl": t.pnl,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "duration_hours": t.duration_seconds / 3600.0 if t.duration_seconds else None,
                "signal_type": t.signal_type,
            }
            for t in sorted_trades[:limit]
        ]
        
        worst = [
            {
                "symbol": t.symbol,
                "pnl": t.pnl,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "duration_hours": t.duration_seconds / 3600.0 if t.duration_seconds else None,
                "signal_type": t.signal_type,
            }
            for t in sorted_trades[-limit:]
        ]
        
        return {"best": best, "worst": worst}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "total_trades": len(self.trades),
            "by_signal_type": self.get_attribution_by_signal_type(),
            "by_symbol": self.get_attribution_by_symbol(),
            "by_regime": self.get_attribution_by_regime(),
            "risk_adjusted": self.get_risk_adjusted_metrics(),
            "best_worst": self.get_best_worst_trades(),
        }
    
    def _save_state(self) -> None:
        """Save performance attribution state."""
        try:
            state_file = Path(self.state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                "trades": [
                    {
                        "symbol": t.symbol,
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "qty": t.qty,
                        "side": t.side,
                        "pnl": t.pnl,
                        "signal_type": t.signal_type,
                        "regime": t.regime,
                        "duration_seconds": t.duration_seconds,
                    }
                    for t in self.trades[-500]  # Keep last 500
                ]
            }
            
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save performance attribution state: %s", e)
    
    def _load_state(self) -> None:
        """Load performance attribution state."""
        try:
            state_file = Path(self.state_file)
            if not state_file.exists():
                return
            
            with state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            for t_data in state_data.get("trades", []):
                trade = TradeRecord(
                    symbol=t_data["symbol"],
                    entry_time=datetime.fromisoformat(t_data["entry_time"]),
                    exit_time=datetime.fromisoformat(t_data["exit_time"]),
                    entry_price=float(t_data["entry_price"]),
                    exit_price=float(t_data["exit_price"]),
                    qty=float(t_data["qty"]),
                    side=t_data["side"],
                    pnl=float(t_data["pnl"]),
                    signal_type=t_data.get("signal_type"),
                    regime=t_data.get("regime"),
                    duration_seconds=t_data.get("duration_seconds"),
                )
                self.trades.append(trade)
        except Exception as e:
            logger.warning("Failed to load performance attribution state: %s", e)
