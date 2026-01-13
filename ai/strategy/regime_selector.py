"""
Regime-Adaptive Strategy Selector
Switch strategies based on market regime
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeights:
    """Strategy weights for a regime."""
    momentum: float = 0.0
    mean_reversion: float = 0.0
    ml: float = 0.0
    volatility: float = 0.0
    defensive: float = 0.0


@dataclass
class RegimeSelectorConfig:
    """Configuration for regime selector."""
    # Strategy mappings by regime
    regime_strategies: Dict[str, StrategyWeights] = field(default_factory=lambda: {
        "RISK_ON": StrategyWeights(momentum=0.4, ml=0.4, mean_reversion=0.2),
        "RISK_OFF": StrategyWeights(defensive=0.5, ml=0.3, momentum=0.2),
        "NEUTRAL": StrategyWeights(ml=0.4, momentum=0.3, mean_reversion=0.3),
        "HIGH_VOL": StrategyWeights(volatility=0.4, ml=0.3, defensive=0.3),
        "trending": StrategyWeights(momentum=0.5, ml=0.3, mean_reversion=0.2),
        "choppy": StrategyWeights(mean_reversion=0.5, ml=0.3, momentum=0.2),
    })


class RegimeSelector:
    """
    Regime-adaptive strategy selector.
    
    Features:
    - Strategy mapping by regime
    - Dynamic strategy weighting
    - Strategy blending during transitions
    """
    
    def __init__(self, config: Optional[RegimeSelectorConfig] = None):
        """
        Initialize regime selector.
        
        Args:
            config: RegimeSelectorConfig instance (optional)
        """
        self.config = config or RegimeSelectorConfig()
        self.current_regime: Optional[str] = None
        
        logger.info("RegimeSelector initialized")
    
    def get_strategy_weights(self, regime: str) -> StrategyWeights:
        """
        Get strategy weights for a regime.
        
        Args:
            regime: Market regime name
        
        Returns:
            StrategyWeights for the regime
        """
        regime = regime.upper()
        
        # Try exact match first
        if regime in self.config.regime_strategies:
            return self.config.regime_strategies[regime]
        
        # Try partial matches
        for key, weights in self.config.regime_strategies.items():
            if key.upper() in regime or regime in key.upper():
                return weights
        
        # Default to neutral
        return self.config.regime_strategies.get("NEUTRAL", StrategyWeights(ml=0.4, momentum=0.3, mean_reversion=0.3))
    
    def apply_strategy_weights(
        self,
        signals: Dict[str, float],
        regime: str,
    ) -> Dict[str, float]:
        """
        Apply strategy weights to signals based on regime.
        
        Args:
            signals: Dict of signal_name -> signal_value
            regime: Current market regime
        
        Returns:
            Weighted signals
        """
        weights = self.get_strategy_weights(regime)
        
        weighted_signals = {}
        
        # Apply weights to each signal type
        if "momentum" in signals:
            weighted_signals["momentum"] = signals["momentum"] * weights.momentum
        if "mean_reversion" in signals or "mean_rev" in signals:
            key = "mean_reversion" if "mean_reversion" in signals else "mean_rev"
            weighted_signals[key] = signals[key] * weights.mean_reversion
        if "ml" in signals:
            weighted_signals["ml"] = signals["ml"] * weights.ml
        if "volatility" in signals:
            weighted_signals["volatility"] = signals["volatility"] * weights.volatility
        if "defensive" in signals:
            weighted_signals["defensive"] = signals["defensive"] * weights.defensive
        
        # Normalize weights
        total_weight = sum([
            weights.momentum, weights.mean_reversion, weights.ml,
            weights.volatility, weights.defensive
        ])
        
        if total_weight > 0:
            for key in weighted_signals:
                weighted_signals[key] = weighted_signals[key] / total_weight
        
        return weighted_signals
