"""
Metrics Collector - Collects and aggregates metrics from bot
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import alpaca_trade_api as tradeapi
from tools.env_loader import ensure_env_loaded

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics from Alpaca account and bot state."""
    
    def __init__(self):
        ensure_env_loaded()
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.client = None
        if self.api_key and self.api_secret:
            try:
                self.client = tradeapi.REST(self.api_key, self.api_secret, self.base_url)
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        try:
            if not self.client:
                return self._get_default_metrics()
            
            account = self.client.get_account()
            positions = self.client.list_positions()
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            cash = float(account.cash)
            portfolio_value = float(account.portfolio_value)
            
            # Calculate PnL
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            
            for pos in positions:
                unrealized_pnl += float(pos.unrealized_pl)
            
            total_pnl = realized_pnl + unrealized_pnl
            
            # Calculate daily PnL (simplified - would need historical data)
            daily_pnl = 0.0  # TODO: Calculate from historical metrics
            
            return {
                "equity": equity,
                "buying_power": buying_power,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "daily_pnl": daily_pnl,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return self._get_default_metrics()
    
    def get_positions(self) -> list:
        """Get current positions."""
        try:
            if not self.client:
                return []
            
            positions = self.client.list_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when API is unavailable."""
        return {
            "equity": 0.0,
            "buying_power": 0.0,
            "cash": 0.0,
            "portfolio_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
