"""
Trade Quality Filter
Only trade when edge is present - professional trade filtering
"""
from __future__ import annotations
import logging
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeQualityConfig:
    """Configuration for trade quality filtering."""
    # Signal validation
    min_expected_value_pct: float = 0.5  # Minimum expected value in %
    min_win_rate: float = 0.45  # Minimum win rate (45%)
    min_signal_strength: float = 0.05  # Minimum signal strength
    
    # Market condition filters
    min_volume: float = 1_000_000.0  # Minimum daily volume
    max_bid_ask_spread_bps: float = 10.0  # Max spread in basis points
    avoid_low_liquidity_hours: bool = True
    low_liquidity_start_hour: int = 15  # 3 PM ET
    low_liquidity_end_hour: int = 16  # 4 PM ET
    
    # Volatility filters
    min_volatility: float = 0.01  # Minimum daily volatility (1%)
    max_volatility: float = 0.10  # Maximum daily volatility (10%)
    
    # Historical performance filters
    min_symbol_win_rate: float = 0.40  # Minimum symbol-specific win rate
    min_strategy_win_rate: float = 0.45  # Minimum strategy-specific win rate
    performance_lookback_days: int = 30  # Days to look back for performance
    
    # State file
    state_file: str = "data/meta/trade_quality.json"


@dataclass
class SignalPerformance:
    """Track performance for a signal type."""
    signal_type: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expected_value: float = 0.0
    last_updated: Optional[datetime] = None


@dataclass
class SymbolPerformance:
    """Track performance for a symbol."""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=20))
    last_updated: Optional[datetime] = None


class TradeQualityFilter:
    """
    Professional trade quality filter.
    
    Features:
    - Signal validation (expected value, win rate)
    - Market condition filters (volume, spread, liquidity)
    - Historical performance filters (symbol/strategy win rates)
    - Real-time quality scoring
    """
    
    def __init__(
        self,
        config: Optional[TradeQualityConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize trade quality filter.
        
        Args:
            config: TradeQualityConfig instance (optional)
            config_path: Path to YAML config file (optional)
        """
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = config or TradeQualityConfig()
        
        # Performance tracking
        self.signal_performance: Dict[str, SignalPerformance] = {}
        self.symbol_performance: Dict[str, SymbolPerformance] = {}
        
        # Recent trades for performance calculation
        self.recent_trades: deque = deque(maxlen=1000)
        
        # Load persisted state
        self._load_state()
        
        logger.info(
            "TradeQualityFilter initialized: min_ev=%.2f%%, min_win_rate=%.2f%%, min_volume=%.0f",
            self.config.min_expected_value_pct,
            self.config.min_win_rate * 100,
            self.config.min_volume,
        )
    
    def _load_config(self, config_path: str) -> TradeQualityConfig:
        """Load configuration from YAML file."""
        import yaml
        path = Path(config_path)
        if not path.exists():
            logger.warning("Trade quality config not found at %s, using defaults", config_path)
            return TradeQualityConfig()
        
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        return TradeQualityConfig(
            min_expected_value_pct=float(raw.get("min_expected_value_pct", 0.5)),
            min_win_rate=float(raw.get("min_win_rate", 0.45)),
            min_signal_strength=float(raw.get("min_signal_strength", 0.05)),
            min_volume=float(raw.get("min_volume", 1_000_000.0)),
            max_bid_ask_spread_bps=float(raw.get("max_bid_ask_spread_bps", 10.0)),
            avoid_low_liquidity_hours=bool(raw.get("avoid_low_liquidity_hours", True)),
            low_liquidity_start_hour=int(raw.get("low_liquidity_start_hour", 15)),
            low_liquidity_end_hour=int(raw.get("low_liquidity_end_hour", 16)),
            min_volatility=float(raw.get("min_volatility", 0.01)),
            max_volatility=float(raw.get("max_volatility", 0.10)),
            min_symbol_win_rate=float(raw.get("min_symbol_win_rate", 0.40)),
            min_strategy_win_rate=float(raw.get("min_strategy_win_rate", 0.45)),
            performance_lookback_days=int(raw.get("performance_lookback_days", 30)),
            state_file=raw.get("state_file", "data/meta/trade_quality.json"),
        )
    
    def filter_signal(
        self,
        symbol: str,
        signal_type: str,
        signal_strength: float,
        price: float,
        volume: Optional[float] = None,
        bid_ask_spread: Optional[float] = None,
        volatility: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Filter a trading signal based on quality criteria.
        
        Args:
            symbol: Symbol
            signal_type: Type of signal (e.g., "momentum", "ml", "mean_rev")
            signal_strength: Signal strength (-1 to 1)
            price: Current price
            volume: Daily volume (optional)
            bid_ask_spread: Bid-ask spread in basis points (optional)
            volatility: Daily volatility (optional)
            market_data: Additional market data dict (optional)
        
        Returns:
            (allowed, reason) tuple - allowed=True if signal passes filter
        """
        # 1. Signal strength check
        if abs(signal_strength) < self.config.min_signal_strength:
            return False, f"signal_strength_too_low ({abs(signal_strength):.4f} < {self.config.min_signal_strength})"
        
        # 2. Market condition filters
        # Volume check
        if volume is not None and volume < self.config.min_volume:
            return False, f"volume_too_low ({volume:,.0f} < {self.config.min_volume:,.0f})"
        
        # Spread check
        if bid_ask_spread is not None and bid_ask_spread > self.config.max_bid_ask_spread_bps:
            return False, f"spread_too_wide ({bid_ask_spread:.2f}bps > {self.config.max_bid_ask_spread_bps:.2f}bps)"
        
        # Time-of-day filter
        if self.config.avoid_low_liquidity_hours:
            now = datetime.now()
            hour = now.hour
            if self.config.low_liquidity_start_hour <= hour < self.config.low_liquidity_end_hour:
                return False, f"low_liquidity_hours ({hour}:00)"
        
        # Volatility filter
        if volatility is not None:
            if volatility < self.config.min_volatility:
                return False, f"volatility_too_low ({volatility:.4f} < {self.config.min_volatility})"
            if volatility > self.config.max_volatility:
                return False, f"volatility_too_high ({volatility:.4f} > {self.config.max_volatility})"
        
        # 3. Historical performance filters
        # Signal type performance
        signal_perf = self.signal_performance.get(signal_type)
        if signal_perf and signal_perf.total_trades >= 10:
            if signal_perf.total_trades > 0:
                win_rate = signal_perf.winning_trades / signal_perf.total_trades
                if win_rate < self.config.min_strategy_win_rate:
                    return False, f"signal_type_win_rate_too_low ({signal_type}: {win_rate:.2%} < {self.config.min_strategy_win_rate:.2%})"
                
                # Expected value check
                if signal_perf.expected_value < self.config.min_expected_value_pct / 100:
                    return False, f"signal_type_ev_too_low ({signal_type}: {signal_perf.expected_value:.4f} < {self.config.min_expected_value_pct/100:.4f})"
        
        # Symbol performance
        symbol_perf = self.symbol_performance.get(symbol)
        if symbol_perf and symbol_perf.total_trades >= 5:
            if symbol_perf.win_rate < self.config.min_symbol_win_rate:
                return False, f"symbol_win_rate_too_low ({symbol}: {symbol_perf.win_rate:.2%} < {self.config.min_symbol_win_rate:.2%})"
        
        # All checks passed
        return True, None
    
    def record_trade_result(
        self,
        symbol: str,
        signal_type: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """
        Record trade result for performance tracking.
        
        Args:
            symbol: Symbol
            signal_type: Type of signal used
            pnl: Profit/loss in dollars
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
        """
        # Record trade
        trade_record = {
            "symbol": symbol,
            "signal_type": signal_type,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
        }
        self.recent_trades.append(trade_record)
        
        # Update signal performance
        if signal_type not in self.signal_performance:
            self.signal_performance[signal_type] = SignalPerformance(signal_type=signal_type)
        
        perf = self.signal_performance[signal_type]
        perf.total_trades += 1
        perf.last_updated = datetime.now()
        
        if pnl > 0:
            perf.winning_trades += 1
            if perf.avg_win == 0:
                perf.avg_win = pnl
            else:
                perf.avg_win = (perf.avg_win * (perf.winning_trades - 1) + pnl) / perf.winning_trades
        else:
            perf.losing_trades += 1
            if perf.avg_loss == 0:
                perf.avg_loss = abs(pnl)
            else:
                perf.avg_loss = (perf.avg_loss * (perf.losing_trades - 1) + abs(pnl)) / perf.losing_trades
        
        perf.total_pnl += pnl
        
        # Calculate expected value
        if perf.total_trades > 0:
            win_rate = perf.winning_trades / perf.total_trades
            loss_rate = perf.losing_trades / perf.total_trades
            if perf.avg_win > 0 and perf.avg_loss > 0:
                perf.expected_value = (win_rate * perf.avg_win - loss_rate * perf.avg_loss) / abs(entry_price - exit_price) if abs(entry_price - exit_price) > 0 else 0.0
        
        # Update symbol performance
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = SymbolPerformance(symbol=symbol)
        
        sym_perf = self.symbol_performance[symbol]
        sym_perf.total_trades += 1
        sym_perf.last_updated = datetime.now()
        
        if pnl > 0:
            sym_perf.winning_trades += 1
        else:
            sym_perf.losing_trades += 1
        
        if sym_perf.total_trades > 0:
            sym_perf.win_rate = sym_perf.winning_trades / sym_perf.total_trades
        
        # Update average PnL
        if sym_perf.avg_pnl == 0:
            sym_perf.avg_pnl = pnl
        else:
            sym_perf.avg_pnl = (sym_perf.avg_pnl * (sym_perf.total_trades - 1) + pnl) / sym_perf.total_trades
        
        # Add to recent performance
        sym_perf.recent_performance.append(1 if pnl > 0 else 0)
        
        # Save state periodically
        if len(self.recent_trades) % 10 == 0:
            self._save_state()
    
    def get_signal_quality_score(
        self,
        symbol: str,
        signal_type: str,
        signal_strength: float,
    ) -> Dict[str, Any]:
        """
        Get quality score for a signal.
        
        Returns:
            Dict with quality metrics and overall score (0-1)
        """
        score = 1.0
        reasons = []
        
        # Signal strength component
        strength_score = min(1.0, abs(signal_strength) / self.config.min_signal_strength)
        score *= strength_score
        
        # Historical performance component
        signal_perf = self.signal_performance.get(signal_type)
        if signal_perf and signal_perf.total_trades >= 10:
            perf_score = signal_perf.winning_trades / max(signal_perf.total_trades, 1)
            score *= perf_score
        else:
            reasons.append("insufficient_signal_history")
        
        # Symbol performance component
        symbol_perf = self.symbol_performance.get(symbol)
        if symbol_perf and symbol_perf.total_trades >= 5:
            symbol_score = symbol_perf.win_rate
            score *= symbol_score
        else:
            reasons.append("insufficient_symbol_history")
        
        return {
            "overall_score": score,
            "strength_score": strength_score,
            "signal_performance": signal_perf.win_rate if signal_perf else None,
            "symbol_performance": symbol_perf.win_rate if symbol_perf else None,
            "reasons": reasons,
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        total_trades = len(self.recent_trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "signal_performance": {},
                "symbol_performance": {},
            }
        
        winning = sum(1 for t in self.recent_trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in self.recent_trades)
        
        return {
            "total_trades": total_trades,
            "win_rate": winning / total_trades if total_trades > 0 else 0.0,
            "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0.0,
            "signal_performance": {
                sig: {
                    "total_trades": perf.total_trades,
                    "win_rate": perf.winning_trades / max(perf.total_trades, 1),
                    "expected_value": perf.expected_value,
                    "avg_win": perf.avg_win,
                    "avg_loss": perf.avg_loss,
                }
                for sig, perf in self.signal_performance.items()
            },
            "symbol_performance": {
                sym: {
                    "total_trades": perf.total_trades,
                    "win_rate": perf.win_rate,
                    "avg_pnl": perf.avg_pnl,
                }
                for sym, perf in self.symbol_performance.items()
            },
        }
    
    def _save_state(self) -> None:
        """Save performance state to file."""
        try:
            state_file = Path(self.config.state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                "signal_performance": {
                    sig: {
                        "total_trades": perf.total_trades,
                        "winning_trades": perf.winning_trades,
                        "losing_trades": perf.losing_trades,
                        "total_pnl": perf.total_pnl,
                        "avg_win": perf.avg_win,
                        "avg_loss": perf.avg_loss,
                        "expected_value": perf.expected_value,
                        "last_updated": perf.last_updated.isoformat() if perf.last_updated else None,
                    }
                    for sig, perf in self.signal_performance.items()
                },
                "symbol_performance": {
                    sym: {
                        "total_trades": perf.total_trades,
                        "winning_trades": perf.winning_trades,
                        "losing_trades": perf.losing_trades,
                        "win_rate": perf.win_rate,
                        "avg_pnl": perf.avg_pnl,
                        "recent_performance": list(perf.recent_performance),
                        "last_updated": perf.last_updated.isoformat() if perf.last_updated else None,
                    }
                    for sym, perf in self.symbol_performance.items()
                },
                "recent_trades": list(self.recent_trades)[-100],  # Keep last 100
            }
            
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save trade quality state: %s", e)
    
    def _load_state(self) -> None:
        """Load performance state from file."""
        try:
            state_file = Path(self.config.state_file)
            if not state_file.exists():
                return
            
            with state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            # Load signal performance
            for sig, data in state_data.get("signal_performance", {}).items():
                self.signal_performance[sig] = SignalPerformance(
                    signal_type=sig,
                    total_trades=int(data.get("total_trades", 0)),
                    winning_trades=int(data.get("winning_trades", 0)),
                    losing_trades=int(data.get("losing_trades", 0)),
                    total_pnl=float(data.get("total_pnl", 0.0)),
                    avg_win=float(data.get("avg_win", 0.0)),
                    avg_loss=float(data.get("avg_loss", 0.0)),
                    expected_value=float(data.get("expected_value", 0.0)),
                    last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
                )
            
            # Load symbol performance
            for sym, data in state_data.get("symbol_performance", {}).items():
                self.symbol_performance[sym] = SymbolPerformance(
                    symbol=sym,
                    total_trades=int(data.get("total_trades", 0)),
                    winning_trades=int(data.get("winning_trades", 0)),
                    losing_trades=int(data.get("losing_trades", 0)),
                    win_rate=float(data.get("win_rate", 0.0)),
                    avg_pnl=float(data.get("avg_pnl", 0.0)),
                    recent_performance=deque(data.get("recent_performance", []), maxlen=20),
                    last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
                )
            
            # Load recent trades
            for trade in state_data.get("recent_trades", []):
                self.recent_trades.append(trade)
        except Exception as e:
            logger.warning("Failed to load trade quality state: %s", e)
