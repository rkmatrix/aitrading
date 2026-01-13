"""
Ticker Manager - Manages trading tickers
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import alpaca_trade_api as tradeapi
from tools.env_loader import ensure_env_loaded

logger = logging.getLogger(__name__)


class TickerManager:
    """Manages ticker configuration and market data."""
    
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
        
        self.config_file = Path("data/ticker_config.json")
        self._load_config()
    
    def _load_config(self):
        """Load ticker configuration."""
        if self.config_file.exists():
            try:
                self.config = json.loads(self.config_file.read_text(encoding="utf-8"))
            except Exception:
                self.config = {"tickers": []}
        else:
            self.config = {"tickers": []}
    
    def _save_config(self):
        """Save ticker configuration."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
    
    def get_tickers(self) -> List[Dict[str, Any]]:
        """Get all configured tickers."""
        tickers = []
        for ticker_config in self.config.get("tickers", []):
            symbol = ticker_config.get("symbol")
            if symbol:
                ticker_data = {
                    "symbol": symbol,
                    "enabled": ticker_config.get("enabled", True),
                    "halted": ticker_config.get("halted", False),
                    "added_at": ticker_config.get("added_at"),
                }
                
                # Get current price if available
                from dashboard.market_data_provider import MarketDataProvider
                provider = MarketDataProvider()
                quote = provider.get_quote(symbol)
                if quote:
                    ticker_data["current_price"] = quote.get('price', 0)
                    ticker_data["price"] = quote.get('price', 0)
                    ticker_data["change_percent"] = quote.get('change_percent', 0)
                    ticker_data["change"] = quote.get('change_percent', 0)
                
                tickers.append(ticker_data)
        
        return tickers
    
    def add_ticker(self, symbol: str) -> Dict[str, Any]:
        """Add ticker to trading list."""
        symbol = symbol.upper().strip()
        
        # Check if already exists
        for ticker in self.config.get("tickers", []):
            if ticker.get("symbol") == symbol:
                return {"success": False, "message": f"{symbol} is already in the list"}
        
        # Validate symbol exists
        if self.client:
            try:
                asset = self.client.get_asset(symbol)
                if not asset.tradable:
                    return {"success": False, "message": f"{symbol} is not tradable"}
            except Exception as e:
                return {"success": False, "message": f"Invalid symbol: {symbol}"}
        
        # Add ticker
        ticker_config = {
            "symbol": symbol,
            "enabled": True,
            "halted": False,
            "added_at": datetime.now().isoformat(),
        }
        
        if "tickers" not in self.config:
            self.config["tickers"] = []
        
        self.config["tickers"].append(ticker_config)
        self._save_config()
        
        return {"success": True, "message": f"{symbol} added successfully"}
    
    def remove_ticker(self, symbol: str) -> Dict[str, Any]:
        """Remove ticker from trading list."""
        symbol = symbol.upper().strip()
        
        tickers = self.config.get("tickers", [])
        original_count = len(tickers)
        
        self.config["tickers"] = [t for t in tickers if t.get("symbol") != symbol]
        
        if len(self.config["tickers"]) < original_count:
            self._save_config()
            return {"success": True, "message": f"{symbol} removed successfully"}
        else:
            return {"success": False, "message": f"{symbol} not found in list"}
    
    def halt_ticker(self, symbol: str) -> Dict[str, Any]:
        """Halt trading for a ticker."""
        symbol = symbol.upper().strip()
        
        for ticker in self.config.get("tickers", []):
            if ticker.get("symbol") == symbol:
                ticker["halted"] = True
                self._save_config()
                return {"success": True, "message": f"{symbol} halted"}
        
        return {"success": False, "message": f"{symbol} not found"}
    
    def resume_ticker(self, symbol: str) -> Dict[str, Any]:
        """Resume trading for a ticker."""
        symbol = symbol.upper().strip()
        
        for ticker in self.config.get("tickers", []):
            if ticker.get("symbol") == symbol:
                ticker["halted"] = False
                self._save_config()
                return {"success": True, "message": f"{symbol} resumed"}
        
        return {"success": False, "message": f"{symbol} not found"}
    
    def search_tickers(self, query: str) -> List[Dict[str, Any]]:
        """Search for tickers."""
        from dashboard.market_data_provider import MarketDataProvider
        
        provider = MarketDataProvider()
        return provider.search_tickers(query, limit=20)
