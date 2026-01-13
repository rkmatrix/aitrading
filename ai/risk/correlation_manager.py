"""
Portfolio Correlation Manager
Prevent over-concentration in correlated assets
"""
from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation management."""
    max_portfolio_correlation: float = 0.7  # Max average correlation
    max_sector_exposure: float = 0.40  # Max 40% per sector
    correlation_window_days: int = 20  # Rolling window for correlation
    min_correlation_samples: int = 10  # Minimum samples for reliable correlation
    
    # Sector mapping
    sector_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "tech": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC"],
        "finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
        "healthcare": ["JNJ", "PFE", "UNH", "ABT", "TMO", "ABBV", "MRK", "LLY"],
        "consumer": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW"],
        "industrial": ["BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT"],
        "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    })
    
    # State file
    state_file: str = "data/meta/correlation_state.json"


class CorrelationManager:
    """
    Portfolio correlation manager.
    
    Features:
    - Real-time correlation tracking
    - Sector exposure limits
    - Dynamic position adjustment
    - Correlation-based position sizing
    """
    
    def __init__(
        self,
        config: Optional[CorrelationConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize correlation manager.
        
        Args:
            config: CorrelationConfig instance (optional)
            config_path: Path to YAML config file (optional)
        """
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = config or CorrelationConfig()
        
        # Price history for correlation calculation
        # symbol -> list of (timestamp, price) tuples
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Correlation cache: (symbol1, symbol2) -> correlation value
        self.correlation_cache: Dict[Tuple[str, str], Tuple[float, datetime]] = {}
        
        # Sector mapping (reverse: symbol -> sector)
        self.symbol_to_sector: Dict[str, str] = {}
        for sector, symbols in self.config.sector_mapping.items():
            for symbol in symbols:
                self.symbol_to_sector[symbol.upper()] = sector
        
        # Load persisted state
        self._load_state()
        
        logger.info(
            "CorrelationManager initialized: max_corr=%.2f, max_sector_exp=%.2f%%, window=%dd",
            self.config.max_portfolio_correlation,
            self.config.max_sector_exposure * 100,
            self.config.correlation_window_days,
        )
    
    def _load_config(self, config_path: str) -> CorrelationConfig:
        """Load configuration from YAML file."""
        import yaml
        path = Path(config_path)
        if not path.exists():
            logger.warning("Correlation config not found at %s, using defaults", config_path)
            return CorrelationConfig()
        
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        sector_mapping = raw.get("sector_mapping", {})
        
        return CorrelationConfig(
            max_portfolio_correlation=float(raw.get("max_portfolio_correlation", 0.7)),
            max_sector_exposure=float(raw.get("max_sector_exposure", 0.40)),
            correlation_window_days=int(raw.get("correlation_window_days", 20)),
            min_correlation_samples=int(raw.get("min_correlation_samples", 10)),
            sector_mapping=sector_mapping,
            state_file=raw.get("state_file", "data/meta/correlation_state.json"),
        )
    
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update price history for a symbol.
        
        Args:
            symbol: Symbol
            price: Current price
            timestamp: Timestamp (defaults to now)
        """
        symbol = symbol.upper()
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append((timestamp, price))
        
        # Keep only recent history (correlation_window_days + buffer)
        cutoff = timestamp - timedelta(days=self.config.correlation_window_days + 5)
        self.price_history[symbol] = [
            (ts, p) for ts, p in self.price_history[symbol] if ts >= cutoff
        ]
    
    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_days: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window_days: Window size (defaults to config value)
        
        Returns:
            Correlation coefficient (-1 to 1) or None if insufficient data
        """
        symbol1 = symbol1.upper()
        symbol2 = symbol2.upper()
        
        if symbol1 == symbol2:
            return 1.0
        
        # Check cache
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self.correlation_cache:
            corr, cache_time = self.correlation_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 3600:  # Cache for 1 hour
                return corr
        
        window_days = window_days or self.config.correlation_window_days
        
        # Get price histories
        hist1 = self.price_history.get(symbol1, [])
        hist2 = self.price_history.get(symbol2, [])
        
        if len(hist1) < self.config.min_correlation_samples or \
           len(hist2) < self.config.min_correlation_samples:
            return None
        
        # Align timestamps and calculate returns
        cutoff = datetime.now() - timedelta(days=window_days)
        hist1_filtered = [(ts, p) for ts, p in hist1 if ts >= cutoff]
        hist2_filtered = [(ts, p) for ts, p in hist2 if ts >= cutoff]
        
        if len(hist1_filtered) < self.config.min_correlation_samples or \
           len(hist2_filtered) < self.config.min_correlation_samples:
            return None
        
        # Create aligned price series
        price_dict1 = {ts: p for ts, p in hist1_filtered}
        price_dict2 = {ts: p for ts, p in hist2_filtered}
        
        common_timestamps = sorted(set(price_dict1.keys()) & set(price_dict2.keys()))
        
        if len(common_timestamps) < self.config.min_correlation_samples:
            return None
        
        prices1 = [price_dict1[ts] for ts in common_timestamps]
        prices2 = [price_dict2[ts] for ts in common_timestamps]
        
        # Calculate returns
        returns1 = np.diff(prices1) / prices1[:-1]
        returns2 = np.diff(prices2) / prices2[:-1]
        
        if len(returns1) < self.config.min_correlation_samples:
            return None
        
        # Calculate correlation
        try:
            corr = float(np.corrcoef(returns1, returns2)[0, 1])
            if np.isnan(corr):
                return None
            
            # Cache result
            self.correlation_cache[cache_key] = (corr, datetime.now())
            
            return corr
        except Exception as e:
            logger.debug("Failed to calculate correlation %s-%s: %s", symbol1, symbol2, e)
            return None
    
    def calculate_portfolio_correlation(
        self,
        positions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level correlation metrics.
        
        Args:
            positions: Dict of symbol -> position weight (0-1)
        
        Returns:
            Dict with correlation metrics
        """
        symbols = [s for s, w in positions.items() if abs(w) > 1e-6]
        
        if len(symbols) < 2:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "correlation_matrix": {},
                "sector_exposure": {},
            }
        
        # Calculate pairwise correlations
        correlations = []
        corr_matrix = {}
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self.calculate_correlation(sym1, sym2)
                if corr is not None:
                    correlations.append(corr)
                    corr_matrix[(sym1, sym2)] = corr
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        max_correlation = max(correlations) if correlations else 0.0
        
        # Calculate sector exposure
        sector_exposure = {}
        for symbol, weight in positions.items():
            sector = self.symbol_to_sector.get(symbol.upper())
            if sector:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(weight)
        
        return {
            "avg_correlation": float(avg_correlation),
            "max_correlation": float(max_correlation),
            "correlation_matrix": corr_matrix,
            "sector_exposure": sector_exposure,
        }
    
    def check_correlation_limits(
        self,
        symbol: str,
        proposed_weight: float,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if adding a position would violate correlation limits.
        
        Args:
            symbol: Symbol to add
            proposed_weight: Proposed position weight (0-1)
            current_positions: Current position weights
        
        Returns:
            (allowed, reason, adjusted_weight) tuple
        """
        symbol = symbol.upper()
        
        # Create hypothetical portfolio
        test_positions = current_positions.copy()
        test_positions[symbol] = proposed_weight
        
        # Check sector exposure
        sector = self.symbol_to_sector.get(symbol)
        if sector:
            sector_exposure = sum(
                abs(w) for s, w in test_positions.items()
                if self.symbol_to_sector.get(s.upper()) == sector
            )
            
            if sector_exposure > self.config.max_sector_exposure:
                # Reduce proposed weight to stay within limit
                current_sector_exposure = sum(
                    abs(w) for s, w in current_positions.items()
                    if self.symbol_to_sector.get(s.upper()) == sector
                )
                max_additional = self.config.max_sector_exposure - current_sector_exposure
                
                if max_additional <= 0:
                    return False, f"sector_exposure_limit ({sector}: {sector_exposure:.2%} > {self.config.max_sector_exposure:.2%})", None
                
                adjusted_weight = min(proposed_weight, max_additional)
                return True, None, adjusted_weight
        
        # Check portfolio correlation
        corr_metrics = self.calculate_portfolio_correlation(test_positions)
        avg_corr = corr_metrics["avg_correlation"]
        
        if avg_corr > self.config.max_portfolio_correlation:
            # Try reducing proposed weight
            # Simple heuristic: reduce by correlation excess
            excess = avg_corr - self.config.max_portfolio_correlation
            reduction_factor = 1.0 - (excess * 2)  # Reduce more if excess is larger
            adjusted_weight = proposed_weight * max(0.1, reduction_factor)
            
            if adjusted_weight < 0.01:  # Too small
                return False, f"portfolio_correlation_too_high (avg_corr={avg_corr:.3f} > {self.config.max_portfolio_correlation:.3f})", None
            
            return True, None, adjusted_weight
        
        return True, None, proposed_weight
    
    def get_diversification_suggestions(
        self,
        current_positions: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions for diversification.
        
        Args:
            current_positions: Current position weights
        
        Returns:
            List of diversification suggestions
        """
        suggestions = []
        
        # Check sector concentration
        sector_exposure = {}
        for symbol, weight in current_positions.items():
            sector = self.symbol_to_sector.get(symbol.upper())
            if sector:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(weight)
        
        # Find underrepresented sectors
        all_sectors = set(self.config.sector_mapping.keys())
        represented_sectors = set(sector_exposure.keys())
        underrepresented = all_sectors - represented_sectors
        
        for sector in underrepresented:
            suggestions.append({
                "type": "sector_diversification",
                "sector": sector,
                "reason": f"No exposure to {sector} sector",
                "suggested_symbols": self.config.sector_mapping.get(sector, [])[:3],
            })
        
        # Check for over-concentration
        for sector, exposure in sector_exposure.items():
            if exposure > self.config.max_sector_exposure:
                suggestions.append({
                    "type": "reduce_concentration",
                    "sector": sector,
                    "reason": f"Sector exposure {exposure:.2%} exceeds limit {self.config.max_sector_exposure:.2%}",
                    "current_exposure": exposure,
                })
        
        return suggestions
    
    def _save_state(self) -> None:
        """Save correlation state to file."""
        try:
            state_file = Path(self.config.state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert price history to serializable format
            price_history_serializable = {
                symbol: [(ts.isoformat(), price) for ts, price in history]
                for symbol, history in self.price_history.items()
            }
            
            # Convert correlation cache
            correlation_cache_serializable = {
                f"{s1}_{s2}": (corr, ts.isoformat())
                for (s1, s2), (corr, ts) in self.correlation_cache.items()
            }
            
            state_data = {
                "price_history": price_history_serializable,
                "correlation_cache": correlation_cache_serializable,
                "last_updated": datetime.now().isoformat(),
            }
            
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save correlation state: %s", e)
    
    def _load_state(self) -> None:
        """Load correlation state from file."""
        try:
            state_file = Path(self.config.state_file)
            if not state_file.exists():
                return
            
            with state_file.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            # Load price history
            for symbol, history in state_data.get("price_history", {}).items():
                self.price_history[symbol] = [
                    (datetime.fromisoformat(ts), float(price))
                    for ts, price in history
                ]
            
            # Load correlation cache
            for key, (corr, ts_str) in state_data.get("correlation_cache", {}).items():
                symbols = key.split("_")
                if len(symbols) == 2:
                    self.correlation_cache[tuple(symbols)] = (
                        float(corr),
                        datetime.fromisoformat(ts_str)
                    )
        except Exception as e:
            logger.warning("Failed to load correlation state: %s", e)
