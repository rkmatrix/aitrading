"""
Adaptive Position Sizing Enhancement
Professional position sizing: Kelly Criterion, volatility targeting, regime-aware sizing
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSizerConfig:
    """Configuration for adaptive position sizing."""
    # Kelly Criterion
    kelly_fraction: float = 0.25  # Fractional Kelly (25% of full Kelly)
    use_kelly: bool = True
    min_kelly_fraction: float = 0.1  # Minimum 10% of Kelly
    max_kelly_fraction: float = 0.5  # Maximum 50% of Kelly
    
    # Volatility targeting
    target_portfolio_volatility: float = 0.12  # 12% annualized
    use_volatility_targeting: bool = True
    
    # Regime-aware sizing
    regime_sizing_enabled: bool = True
    high_vol_reduction: float = 0.5  # Reduce size by 50% in high vol
    trending_increase: float = 1.2  # Increase size by 20% in trending regimes
    choppy_reduction: float = 0.3  # Reduce to 30% in choppy regimes
    
    # Edge-based sizing
    min_edge_threshold: float = 0.01  # Minimum 1% edge to trade
    edge_scaling_factor: float = 2.0  # Scale size by edge


class AdaptiveSizer:
    """
    Professional adaptive position sizer.
    
    Features:
    - Kelly Criterion implementation
    - Volatility targeting
    - Regime-aware sizing
    - Edge-based sizing
    """
    
    def __init__(self, config: Optional[AdaptiveSizerConfig] = None):
        """
        Initialize adaptive sizer.
        
        Args:
            config: AdaptiveSizerConfig instance (optional)
        """
        self.config = config or AdaptiveSizerConfig()
        
        logger.info(
            "AdaptiveSizer initialized: kelly_fraction=%.2f, target_vol=%.2f%%, regime_sizing=%s",
            self.config.kelly_fraction,
            self.config.target_portfolio_volatility * 100,
            self.config.regime_sizing_enabled,
        )
    
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            equity: Account equity
        
        Returns:
            Position size in dollars
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly percentage = (win_rate * avg_win - loss_rate * avg_loss) / avg_loss
        loss_rate = 1.0 - win_rate
        kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_loss
        
        # Clamp Kelly to reasonable range
        kelly_pct = max(0.0, min(kelly_pct, 0.25))  # Max 25% per position
        
        # Apply fractional Kelly
        fractional_kelly = kelly_pct * self.config.kelly_fraction
        
        # Clamp to min/max bounds
        fractional_kelly = max(
            self.config.min_kelly_fraction * kelly_pct,
            min(fractional_kelly, self.config.max_kelly_fraction * kelly_pct)
        )
        
        position_size = equity * fractional_kelly
        
        return position_size
    
    def calculate_volatility_targeted_size(
        self,
        symbol_volatility: float,
        portfolio_volatility: float,
        base_size: float,
        target_volatility: Optional[float] = None,
    ) -> float:
        """
        Adjust position size to target portfolio volatility.
        
        Args:
            symbol_volatility: Symbol's volatility (annualized)
            portfolio_volatility: Current portfolio volatility (annualized)
            base_size: Base position size
            target_volatility: Target portfolio volatility (defaults to config)
        
        Returns:
            Adjusted position size
        """
        if not self.config.use_volatility_targeting:
            return base_size
        
        target_vol = target_volatility or self.config.target_portfolio_volatility
        
        if symbol_volatility <= 0 or portfolio_volatility <= 0:
            return base_size
        
        # Calculate volatility contribution
        # Simple heuristic: adjust size inversely to volatility
        vol_ratio = target_vol / max(symbol_volatility, 0.01)
        
        # Clamp adjustment
        vol_ratio = max(0.5, min(vol_ratio, 2.0))  # Between 50% and 200%
        
        adjusted_size = base_size * vol_ratio
        
        return adjusted_size
    
    def apply_regime_sizing(
        self,
        base_size: float,
        regime: str,
    ) -> float:
        """
        Adjust position size based on market regime.
        
        Args:
            base_size: Base position size
            regime: Market regime ("trending", "choppy", "high_vol", "normal")
        
        Returns:
            Regime-adjusted position size
        """
        if not self.config.regime_sizing_enabled:
            return base_size
        
        regime = regime.lower()
        
        if regime in ["high_vol", "volatile", "chaos"]:
            return base_size * (1.0 - self.config.high_vol_reduction)
        elif regime in ["trending", "bull", "bear"]:
            return base_size * self.config.trending_increase
        elif regime in ["choppy", "rangebound", "sideways"]:
            return base_size * self.config.choppy_reduction
        else:
            return base_size  # Normal regime, no adjustment
    
    def calculate_edge_based_size(
        self,
        base_size: float,
        expected_value: float,
        signal_strength: float,
    ) -> float:
        """
        Adjust position size based on expected edge.
        
        Args:
            base_size: Base position size
            expected_value: Expected value (win_rate * avg_win - loss_rate * avg_loss)
            signal_strength: Signal strength (-1 to 1)
        
        Returns:
            Edge-adjusted position size
        """
        # Calculate edge percentage
        edge_pct = abs(expected_value) if expected_value != 0 else abs(signal_strength)
        
        if edge_pct < self.config.min_edge_threshold:
            return 0.0  # No edge, don't trade
        
        # Scale by edge
        edge_multiplier = 1.0 + (edge_pct - self.config.min_edge_threshold) * self.config.edge_scaling_factor
        
        # Clamp multiplier
        edge_multiplier = max(0.5, min(edge_multiplier, 2.0))
        
        return base_size * edge_multiplier
    
    def calculate_optimal_size(
        self,
        symbol: str,
        signal_strength: float,
        equity: float,
        symbol_volatility: Optional[float] = None,
        portfolio_volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        expected_value: Optional[float] = None,
        regime: Optional[str] = None,
        base_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using all available methods.
        
        Args:
            symbol: Symbol
            signal_strength: Signal strength (-1 to 1)
            equity: Account equity
            symbol_volatility: Symbol volatility (optional)
            portfolio_volatility: Portfolio volatility (optional)
            win_rate: Historical win rate (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            expected_value: Expected value (optional)
            regime: Market regime (optional)
            base_size: Base size to adjust (optional)
        
        Returns:
            Dict with size calculation details
        """
        # Start with base size or calculate from signal strength
        if base_size is None:
            # Simple base size: 2% of equity per unit of signal strength
            base_size = equity * 0.02 * abs(signal_strength)
        else:
            base_size = float(base_size)
        
        size = base_size
        adjustments = []
        
        # 1. Kelly Criterion (if we have win rate data)
        if self.config.use_kelly and win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_size = self.calculate_kelly_size(win_rate, avg_win, avg_loss, equity)
            if kelly_size > 0:
                # Blend Kelly with base size
                size = (size + kelly_size) / 2.0
                adjustments.append(f"kelly={kelly_size:.2f}")
        
        # 2. Volatility targeting
        if symbol_volatility is not None and portfolio_volatility is not None:
            size = self.calculate_volatility_targeted_size(
                symbol_volatility,
                portfolio_volatility,
                size,
            )
            adjustments.append(f"vol_targeting")
        
        # 3. Regime-aware sizing
        if regime:
            size = self.apply_regime_sizing(size, regime)
            adjustments.append(f"regime={regime}")
        
        # 4. Edge-based sizing
        if expected_value is not None:
            size = self.calculate_edge_based_size(size, expected_value, signal_strength)
            if size == 0:
                adjustments.append("no_edge")
            else:
                adjustments.append(f"edge_based")
        
        # Final clamp: ensure size doesn't exceed reasonable limits
        max_size = equity * 0.25  # Max 25% of equity per position
        size = min(size, max_size)
        
        return {
            "symbol": symbol,
            "optimal_size": size,
            "base_size": base_size,
            "size_as_pct_of_equity": size / equity if equity > 0 else 0.0,
            "adjustments": adjustments,
            "kelly_size": kelly_size if (win_rate and avg_win and avg_loss) else None,
        }
