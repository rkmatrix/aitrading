"""
Enhanced Market Data Provider
Supports multiple free data sources with automatic fallback
"""
from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available - market data fallback disabled")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class EnhancedMarketDataProvider:
    """
    Multi-source market data provider with automatic fallback.
    
    Priority order:
    1. Alpaca API (if credentials available)
    2. yfinance (free, reliable)
    3. Alpha Vantage (free tier, requires API key)
    4. Polygon.io (free tier, requires API key)
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
        self.rate_limit_delay = 0.5  # Minimum delay between requests
        
        # Initialize Alpaca if available
        self.alpaca_client = None
        if ALPACA_AVAILABLE:
            try:
                api_key = os.getenv("APCA_API_KEY_ID")
                api_secret = os.getenv("APCA_API_SECRET_KEY")
                base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
                
                if api_key and api_secret:
                    self.alpaca_client = tradeapi.REST(api_key, api_secret, base_url)
                    logger.info("✅ Alpaca client initialized for market data")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca client: {e}")
        
        # Alpha Vantage API key (optional)
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        # Polygon.io API key (optional)
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = (data, time.time())
    
    def _rate_limit(self):
        """Simple rate limiting."""
        time.sleep(self.rate_limit_delay)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Get last trade price for a symbol.
        Tries multiple sources with automatic fallback.
        """
        cache_key = f"price_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Try Alpaca first (if available)
        if self.alpaca_client:
            try:
                self._rate_limit()
                quote = self.alpaca_client.get_latest_quote(symbol)
                if quote and hasattr(quote, 'bp') and quote.bp:
                    price = float(quote.bp)  # Use bid price
                    self._set_cache(cache_key, price)
                    return price
            except Exception as e:
                logger.debug(f"Alpaca price fetch failed for {symbol}: {e}")
        
        # Try yfinance (free, reliable)
        if YFINANCE_AVAILABLE:
            try:
                self._rate_limit()
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
                    self._set_cache(cache_key, price)
                    return price
            except Exception as e:
                logger.debug(f"yfinance price fetch failed for {symbol}: {e}")
        
        # Try Alpha Vantage (free tier: 5 calls/min, 500/day)
        if self.alpha_vantage_key and REQUESTS_AVAILABLE:
            try:
                self._rate_limit()
                url = f"https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_key
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    quote = data.get("Global Quote", {})
                    if quote:
                        price_str = quote.get("05. price")
                        if price_str:
                            price = float(price_str)
                            self._set_cache(cache_key, price)
                            return price
            except Exception as e:
                logger.debug(f"Alpha Vantage price fetch failed for {symbol}: {e}")
        
        # Try Polygon.io (free tier available)
        if self.polygon_key and REQUESTS_AVAILABLE:
            try:
                self._rate_limit()
                url = f"https://api.polygon.io/v2/last/trade/{symbol}"
                headers = {"Authorization": f"Bearer {self.polygon_key}"}
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "OK":
                        price = float(data["results"]["p"])
                        self._set_cache(cache_key, price)
                        return price
            except Exception as e:
                logger.debug(f"Polygon.io price fetch failed for {symbol}: {e}")
        
        logger.warning(f"⚠️ All price sources failed for {symbol}")
        return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data.
        Returns list of dicts with: time, open, high, low, close, volume
        """
        cache_key = f"history_{symbol}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Try yfinance first (best free option)
        if YFINANCE_AVAILABLE:
            try:
                self._rate_limit()
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    bars = []
                    for idx, row in data.iterrows():
                        bars.append({
                            "time": int(idx.timestamp() * 1000),  # milliseconds
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": int(row["Volume"]),
                        })
                    self._set_cache(cache_key, bars)
                    return bars
            except Exception as e:
                logger.debug(f"yfinance historical data failed for {symbol}: {e}")
        
        # Try Alpaca as fallback
        if self.alpaca_client:
            try:
                self._rate_limit()
                # Map period to Alpaca timeframe
                timeframe_map = {
                    "1d": "1Min",
                    "1w": "5Min",
                    "1m": "1Day",
                    "3m": "1Day",
                    "6m": "1Day",
                    "1y": "1Day",
                }
                timeframe = timeframe_map.get(interval, "1Day")
                
                bars = self.alpaca_client.get_bars(
                    symbol,
                    timeframe,
                    limit=100
                ).df
                
                if not bars.empty:
                    result = []
                    for idx, row in bars.iterrows():
                        result.append({
                            "time": int(idx.timestamp() * 1000),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": int(row["volume"]),
                        })
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.debug(f"Alpaca historical data failed for {symbol}: {e}")
        
        logger.warning(f"⚠️ Historical data fetch failed for {symbol}")
        return []
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current quote (bid, ask, last price).
        Returns dict with: symbol, price, bid, ask, change, change_percent
        """
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Try Alpaca first
        if self.alpaca_client:
            try:
                self._rate_limit()
                quote = self.alpaca_client.get_latest_quote(symbol)
                if quote:
                    result = {
                        "symbol": symbol,
                        "bid": float(quote.bp) if quote.bp else None,
                        "ask": float(quote.ap) if quote.ap else None,
                        "price": float(quote.bp) if quote.bp else None,
                    }
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.debug(f"Alpaca quote failed for {symbol}: {e}")
        
        # Try yfinance
        if YFINANCE_AVAILABLE:
            try:
                self._rate_limit()
                ticker = yf.Ticker(symbol)
                info = ticker.info
                quote_data = ticker.history(period="1d", interval="1m")
                
                if not quote_data.empty:
                    current_price = float(quote_data['Close'].iloc[-1])
                    prev_close = info.get("previousClose", current_price)
                    change = current_price - prev_close
                    change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    result = {
                        "symbol": symbol,
                        "price": current_price,
                        "bid": current_price * 0.9999,  # Approximate
                        "ask": current_price * 1.0001,  # Approximate
                        "change": change,
                        "change_percent": change_percent,
                    }
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.debug(f"yfinance quote failed for {symbol}: {e}")
        
        return None
    
    def search_tickers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for tickers by name or symbol.
        Uses yfinance for free ticker search.
        """
        if not YFINANCE_AVAILABLE:
            return []
        
        try:
            # yfinance doesn't have a search API, but we can try common patterns
            # For production, consider using a dedicated search API
            results = []
            
            # Try direct symbol lookup
            try:
                ticker = yf.Ticker(query.upper())
                info = ticker.info
                if info and info.get("symbol"):
                    results.append({
                        "symbol": info["symbol"],
                        "name": info.get("longName", info.get("shortName", "")),
                        "exchange": info.get("exchange", ""),
                        "type": info.get("quoteType", ""),
                    })
            except Exception:
                pass
            
            return results[:limit]
        except Exception as e:
            logger.debug(f"Ticker search failed: {e}")
            return []
