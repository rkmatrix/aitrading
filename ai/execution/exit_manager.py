"""
Advanced Exit Management System
Professional exit strategies: trailing stops, profit targets, time-based exits
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """Configuration for exit strategies."""
    # Trailing stops
    trailing_stop_enabled: bool = True
    trailing_stop_method: str = "atr"  # "atr" or "percentage"
    trailing_stop_atr_multiplier: float = 2.5
    trailing_stop_percentage: float = 0.02  # 2%
    breakeven_enabled: bool = True
    breakeven_after_pct: float = 0.01  # Move to breakeven after 1% profit
    
    # Profit targets
    profit_targets_enabled: bool = True
    profit_targets: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])  # Partial exits
    risk_reward_ratio: float = 2.0  # Target profit = risk * ratio
    
    # Time-based exits
    time_based_exit_enabled: bool = True
    max_hold_hours: float = 24.0
    exit_on_signal_reversal: bool = True
    exit_on_regime_change: bool = True
    exit_on_volatility_spike: bool = True
    
    # ATR calculation
    atr_period: int = 14
    
    # State persistence
    state_file: str = "data/runtime/exit_manager_state.json"


@dataclass
class PositionExitState:
    """Tracks exit state for a single position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    current_qty: float  # Signed: positive for long, negative for short
    side: str  # "LONG" or "SHORT"
    
    # Trailing stop
    trailing_stop_price: Optional[float] = None
    highest_price: Optional[float] = None  # For longs
    lowest_price: Optional[float] = None  # For shorts
    breakeven_triggered: bool = False
    
    # Profit targets
    profit_targets_hit: List[float] = field(default_factory=list)  # Which targets already hit
    initial_risk: Optional[float] = None  # Entry price - stop loss (for risk/reward)
    
    # Time tracking
    hours_held: float = 0.0
    
    # Exit signals
    exit_signal_active: bool = False
    exit_reason: Optional[str] = None


class ExitManager:
    """
    Professional exit management system.
    
    Features:
    - ATR-based and percentage-based trailing stops
    - Multiple profit targets with partial exits
    - Time-based exits
    - Signal reversal exits
    - Regime change exits
    """
    
    def __init__(self, config: Optional[ExitConfig] = None, config_path: Optional[str] = None):
        """
        Initialize exit manager.
        
        Args:
            config: ExitConfig instance (optional)
            config_path: Path to YAML config file (optional)
        """
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = config or ExitConfig()
        
        # Position state tracking: symbol -> PositionExitState
        self.position_states: Dict[str, PositionExitState] = {}
        
        # ATR cache: symbol -> (atr_value, last_update_time)
        self._atr_cache: Dict[str, Tuple[float, float]] = {}
        
        # Load persisted state
        self._load_state()
        
        logger.info(
            "ExitManager initialized: trailing_stops=%s, profit_targets=%s, time_based=%s",
            self.config.trailing_stop_enabled,
            self.config.profit_targets_enabled,
            self.config.time_based_exit_enabled,
        )
    
    def _load_config(self, config_path: str) -> ExitConfig:
        """Load configuration from YAML file."""
        import yaml
        path = Path(config_path)
        if not path.exists():
            logger.warning("Exit config not found at %s, using defaults", config_path)
            return ExitConfig()
        
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        trailing = raw.get("trailing_stops", {})
        profit = raw.get("profit_targets", {})
        time_exit = raw.get("time_based_exits", {})
        
        return ExitConfig(
            trailing_stop_enabled=trailing.get("enabled", True),
            trailing_stop_method=trailing.get("method", "atr"),
            trailing_stop_atr_multiplier=float(trailing.get("atr_multiplier", 2.5)),
            trailing_stop_percentage=float(trailing.get("percentage", 0.02)),
            breakeven_enabled=trailing.get("breakeven_enabled", True),
            breakeven_after_pct=float(trailing.get("breakeven_after_pct", 0.01)),
            profit_targets_enabled=profit.get("enabled", True),
            profit_targets=profit.get("targets", [0.25, 0.50, 0.75]),
            risk_reward_ratio=float(profit.get("risk_reward_ratio", 2.0)),
            time_based_exit_enabled=time_exit.get("enabled", True),
            max_hold_hours=float(time_exit.get("max_hold_hours", 24.0)),
            exit_on_signal_reversal=time_exit.get("exit_on_signal_reversal", True),
            exit_on_regime_change=time_exit.get("exit_on_regime_change", True),
            exit_on_volatility_spike=time_exit.get("exit_on_volatility_spike", True),
            state_file=raw.get("state_file", "data/runtime/exit_manager_state.json"),
        )
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        entry_time: Optional[datetime] = None,
        initial_stop_loss: Optional[float] = None,
    ) -> None:
        """
        Register a new position for exit management.
        
        Args:
            symbol: Symbol
            entry_price: Entry price
            qty: Position quantity (positive for long, negative for short)
            entry_time: Entry timestamp (defaults to now)
            initial_stop_loss: Initial stop loss price (for risk/reward calculation)
        """
        side = "LONG" if qty > 0 else "SHORT"
        entry_time = entry_time or datetime.now()
        
        # Calculate initial risk
        initial_risk = None
        if initial_stop_loss:
            if side == "LONG":
                initial_risk = entry_price - initial_stop_loss
            else:
                initial_risk = initial_stop_loss - entry_price
        
        state = PositionExitState(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            current_qty=qty,
            side=side,
            highest_price=entry_price if side == "LONG" else None,
            lowest_price=entry_price if side == "SHORT" else None,
            initial_risk=initial_risk,
        )
        
        self.position_states[symbol] = state
        logger.info(
            "Registered position: %s %s @ %.2f, qty=%.2f",
            symbol, side, entry_price, qty
        )
        self._save_state()
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_qty: Optional[float] = None,
        atr: Optional[float] = None,
        signal_reversed: bool = False,
        regime_changed: bool = False,
        volatility_spike: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Update position and check for exit signals.
        
        Args:
            symbol: Symbol
            current_price: Current market price
            current_qty: Current position quantity (if changed)
            atr: Current ATR value (for trailing stops)
            signal_reversed: Whether signal has reversed
            regime_changed: Whether market regime changed
            volatility_spike: Whether volatility spiked
        
        Returns:
            Exit order dict if exit triggered, None otherwise
            Format: {"symbol": str, "side": "SELL"/"BUY", "qty": float, "reason": str}
        """
        if symbol not in self.position_states:
            return None
        
        state = self.position_states[symbol]
        
        # Update quantity if provided
        if current_qty is not None:
            state.current_qty = current_qty
            # If position closed, remove from tracking
            if abs(current_qty) < 1e-6:
                del self.position_states[symbol]
                self._save_state()
                return None
        
        # Update time held
        hours_held = (datetime.now() - state.entry_time).total_seconds() / 3600.0
        state.hours_held = hours_held
        
        # Update highest/lowest prices
        if state.side == "LONG":
            if state.highest_price is None or current_price > state.highest_price:
                state.highest_price = current_price
        else:  # SHORT
            if state.lowest_price is None or current_price < state.lowest_price:
                state.lowest_price = current_price
        
        # Check exit signals
        exit_order = None
        
        # 1. Trailing stop check
        if self.config.trailing_stop_enabled:
            exit_order = self._check_trailing_stop(state, current_price, atr)
            if exit_order:
                return exit_order
        
        # 2. Profit target check
        if self.config.profit_targets_enabled and not exit_order:
            exit_order = self._check_profit_targets(state, current_price)
            if exit_order:
                return exit_order
        
        # 3. Time-based exit
        if self.config.time_based_exit_enabled and not exit_order:
            exit_order = self._check_time_based_exit(state)
            if exit_order:
                return exit_order
        
        # 4. Signal reversal exit
        if self.config.exit_on_signal_reversal and signal_reversed and not exit_order:
            exit_order = self._create_exit_order(state, "signal_reversal")
            return exit_order
        
        # 5. Regime change exit
        if self.config.exit_on_regime_change and regime_changed and not exit_order:
            exit_order = self._create_exit_order(state, "regime_change")
            return exit_order
        
        # 6. Volatility spike exit
        if self.config.exit_on_volatility_spike and volatility_spike and not exit_order:
            exit_order = self._create_exit_order(state, "volatility_spike")
            return exit_order
        
        # Update trailing stop price even if no exit
        if self.config.trailing_stop_enabled:
            self._update_trailing_stop(state, current_price, atr)
        
        self._save_state()
        return None
    
    def _check_trailing_stop(
        self,
        state: PositionExitState,
        current_price: float,
        atr: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Check if trailing stop is triggered."""
        if state.side == "LONG":
            # Long position: stop below current price
            if state.trailing_stop_price and current_price <= state.trailing_stop_price:
                return self._create_exit_order(state, "trailing_stop")
        else:
            # Short position: stop above current price
            if state.trailing_stop_price and current_price >= state.trailing_stop_price:
                return self._create_exit_order(state, "trailing_stop")
        
        return None
    
    def _update_trailing_stop(
        self,
        state: PositionExitState,
        current_price: float,
        atr: Optional[float],
    ) -> None:
        """Update trailing stop price."""
        if state.side == "LONG":
            # Calculate profit percentage
            profit_pct = (current_price - state.entry_price) / state.entry_price
            
            # Breakeven logic
            if self.config.breakeven_enabled and not state.breakeven_triggered:
                if profit_pct >= self.config.breakeven_after_pct:
                    state.trailing_stop_price = state.entry_price
                    state.breakeven_triggered = True
                    logger.debug("%s: Trailing stop moved to breakeven", state.symbol)
            
            # Update trailing stop
            if state.highest_price:
                if self.config.trailing_stop_method == "atr" and atr:
                    stop_distance = atr * self.config.trailing_stop_atr_multiplier
                    new_stop = state.highest_price - stop_distance
                else:
                    stop_distance = state.highest_price * self.config.trailing_stop_percentage
                    new_stop = state.highest_price - stop_distance
                
                # Only move stop up, never down
                if state.trailing_stop_price is None or new_stop > state.trailing_stop_price:
                    state.trailing_stop_price = new_stop
        
        else:  # SHORT
            # Calculate profit percentage
            profit_pct = (state.entry_price - current_price) / state.entry_price
            
            # Breakeven logic
            if self.config.breakeven_enabled and not state.breakeven_triggered:
                if profit_pct >= self.config.breakeven_after_pct:
                    state.trailing_stop_price = state.entry_price
                    state.breakeven_triggered = True
                    logger.debug("%s: Trailing stop moved to breakeven", state.symbol)
            
            # Update trailing stop
            if state.lowest_price:
                if self.config.trailing_stop_method == "atr" and atr:
                    stop_distance = atr * self.config.trailing_stop_atr_multiplier
                    new_stop = state.lowest_price + stop_distance
                else:
                    stop_distance = state.lowest_price * self.config.trailing_stop_percentage
                    new_stop = state.lowest_price + stop_distance
                
                # Only move stop down, never up
                if state.trailing_stop_price is None or new_stop < state.trailing_stop_price:
                    state.trailing_stop_price = new_stop
    
    def _check_profit_targets(
        self,
        state: PositionExitState,
        current_price: float,
    ) -> Optional[Dict[str, Any]]:
        """Check if profit targets are hit."""
        if state.side == "LONG":
            profit_pct = (current_price - state.entry_price) / state.entry_price
        else:
            profit_pct = (state.entry_price - current_price) / state.entry_price
        
        # Check each profit target
        for target_pct in self.config.profit_targets:
            if target_pct in state.profit_targets_hit:
                continue  # Already hit this target
            
            # Calculate target price
            if state.side == "LONG":
                target_price = state.entry_price * (1 + target_pct)
                if current_price >= target_price:
                    state.profit_targets_hit.append(target_pct)
                    # Partial exit: exit the target percentage of position
                    exit_qty = abs(state.current_qty) * target_pct
                    return {
                        "symbol": state.symbol,
                        "side": "SELL" if state.side == "LONG" else "BUY",
                        "qty": exit_qty,
                        "reason": f"profit_target_{int(target_pct * 100)}%",
                        "partial": True,
                    }
            else:  # SHORT
                target_price = state.entry_price * (1 - target_pct)
                if current_price <= target_price:
                    state.profit_targets_hit.append(target_pct)
                    exit_qty = abs(state.current_qty) * target_pct
                    return {
                        "symbol": state.symbol,
                        "side": "BUY",
                        "qty": exit_qty,
                        "reason": f"profit_target_{int(target_pct * 100)}%",
                        "partial": True,
                    }
        
        return None
    
    def _check_time_based_exit(self, state: PositionExitState) -> Optional[Dict[str, Any]]:
        """Check if time-based exit is triggered."""
        if state.hours_held >= self.config.max_hold_hours:
            return self._create_exit_order(state, f"max_hold_time_{self.config.max_hold_hours}h")
        return None
    
    def _create_exit_order(
        self,
        state: PositionExitState,
        reason: str,
    ) -> Dict[str, Any]:
        """Create exit order dict."""
        state.exit_signal_active = True
        state.exit_reason = reason
        
        return {
            "symbol": state.symbol,
            "side": "SELL" if state.side == "LONG" else "BUY",
            "qty": abs(state.current_qty),
            "reason": reason,
            "partial": False,
        }
    
    def get_position_exit_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get exit information for a position."""
        if symbol not in self.position_states:
            return None
        
        state = self.position_states[symbol]
        
        return {
            "symbol": symbol,
            "entry_price": state.entry_price,
            "entry_time": state.entry_time.isoformat(),
            "trailing_stop_price": state.trailing_stop_price,
            "profit_targets_hit": state.profit_targets_hit,
            "hours_held": state.hours_held,
            "exit_signal_active": state.exit_signal_active,
            "exit_reason": state.exit_reason,
        }
    
    def get_all_positions_info(self) -> Dict[str, Dict[str, Any]]:
        """Get exit info for all tracked positions."""
        return {
            symbol: self.get_position_exit_info(symbol)
            for symbol in self.position_states.keys()
        }
    
    def _save_state(self) -> None:
        """Save position states to file."""
        try:
            state_file = Path(self.config.state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {}
            for symbol, state in self.position_states.items():
                state_data[symbol] = {
                    "entry_price": state.entry_price,
                    "entry_time": state.entry_time.isoformat(),
                    "current_qty": state.current_qty,
                    "side": state.side,
                    "trailing_stop_price": state.trailing_stop_price,
                    "highest_price": state.highest_price,
                    "lowest_price": state.lowest_price,
                    "breakeven_triggered": state.breakeven_triggered,
                    "profit_targets_hit": state.profit_targets_hit,
                    "initial_risk": state.initial_risk,
                    "hours_held": state.hours_held,
                    "exit_signal_active": state.exit_signal_active,
                    "exit_reason": state.exit_reason,
                }
            
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save exit manager state: %s", e)
    
    def _load_state(self) -> None:
        """Load position states from file."""
        try:
            state_file = Path(self.config.state_file)
            if not state_file.exists():
                return
            
            with state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            for symbol, data in state_data.items():
                try:
                    state = PositionExitState(
                        symbol=symbol,
                        entry_price=float(data["entry_price"]),
                        entry_time=datetime.fromisoformat(data["entry_time"]),
                        current_qty=float(data["current_qty"]),
                        side=data["side"],
                        trailing_stop_price=data.get("trailing_stop_price"),
                        highest_price=data.get("highest_price"),
                        lowest_price=data.get("lowest_price"),
                        breakeven_triggered=bool(data.get("breakeven_triggered", False)),
                        profit_targets_hit=list(data.get("profit_targets_hit", [])),
                        initial_risk=data.get("initial_risk"),
                        hours_held=float(data.get("hours_held", 0.0)),
                        exit_signal_active=bool(data.get("exit_signal_active", False)),
                        exit_reason=data.get("exit_reason"),
                    )
                    self.position_states[symbol] = state
                except Exception as e:
                    logger.warning("Failed to load state for %s: %s", symbol, e)
        except Exception as e:
            logger.warning("Failed to load exit manager state: %s", e)
    
    def calculate_atr(
        self,
        symbol: str,
        prices: pd.DataFrame,
        period: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate ATR from price data.
        
        Args:
            symbol: Symbol
            prices: DataFrame with columns ['high', 'low', 'close']
            period: ATR period (defaults to config value)
        
        Returns:
            ATR value or None if insufficient data
        """
        period = period or self.config.atr_period
        
        if len(prices) < period + 1:
            return None
        
        try:
            high = prices['high'].values
            low = prices['low'].values
            close = prices['close'].values
            
            # True Range calculation
            tr_list = []
            for i in range(1, len(prices)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr_list.append(max(tr1, tr2, tr3))
            
            # ATR = SMA of True Range
            atr = np.mean(tr_list[-period:])
            
            # Cache ATR
            self._atr_cache[symbol] = (atr, time.time())
            
            return float(atr)
        except Exception as e:
            logger.warning("Failed to calculate ATR for %s: %s", symbol, e)
            return None
