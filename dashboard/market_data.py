"""
Market Data Integration
"""
import os
import logging
from typing import Dict, Any, Optional
import alpaca_trade_api as tradeapi
from tools.env_loader import ensure_env_loaded

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Provides market data from Alpaca."""
    
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
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for a symbol."""
        if not self.client:
            return None
        
        try:
            quote = self.client.get_latest_quote(symbol)
            return {
                "symbol": symbol,
                "bid": float(quote.bp) if quote.bp else None,
                "ask": float(quote.ap) if quote.ap else None,
                "bid_size": int(quote.bs) if quote.bs else None,
                "ask_size": int(quote.as_) if quote.as_ else None,
            }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> list:
        """Get historical bars for a symbol."""
        if not self.client:
            return []
        
        try:
            bars = self.client.get_bars(symbol, timeframe, limit=limit).df
            return [
                {
                    "time": bar.name.timestamp(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
                for bar in bars.itertuples()
            ]
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            return []
