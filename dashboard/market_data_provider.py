"""
Unified Market Data Provider
Supports multiple free data sources with fallback mechanisms
"""
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Unified market data provider with multiple sources and caching."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
        self.last_request_time = {}
        self.min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        
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
        now = time.time()
        if 'last_request' in self.last_request_time:
            elapsed = now - self.last_request_time['last_request']
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time['last_request'] = time.time()
    
    def search_tickers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for tickers by symbol or company name."""
        cache_key = f"search_{query.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached[:limit]
        
        self._rate_limit()
        query = query.upper().strip()
        results = []
        
        try:
            # Use yfinance to search - it has a ticker list we can search
            # For now, we'll use a common stocks list and filter
            # In production, you might want to use a proper ticker database
            
            # Common US stocks for search - expanded list
            common_tickers = [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'AMD',
                'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM', 'NOW', 'SNOW', 'PLTR', 'CRWD', 'ZS',
                'PANW', 'NET', 'DDOG', 'MDB', 'MELI', 'SHOP', 'SQ', 'PYPL', 'HOOD', 'COIN',
                # Finance
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'BLK', 'AXP', 'V', 'MA',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'GILD',
                'AMGN', 'BIIB', 'REGN', 'VRTX', 'ZTS', 'CI', 'HUM', 'CVS', 'ELV',
                # Consumer
                'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'YUM', 'CMG',
                'DIS', 'NFLX', 'PARA', 'WBD', 'RBLX', 'TTWO', 'EA', 'ATVI',
                # Industrial
                'BA', 'CAT', 'DE', 'GE', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'TXT',
                'EMR', 'ETN', 'ITW', 'PH', 'AME', 'ROK', 'DOV',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'OXY', 'DVN',
                # Materials
                'LIN', 'APD', 'SHW', 'PPG', 'ECL', 'DD', 'DOW', 'FCX', 'NEM',
                # Utilities
                'NEE', 'DUK', 'SO', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG',
                # Real Estate
                'AMT', 'PLD', 'EQIX', 'PSA', 'WELL', 'VICI', 'SPG', 'O',
                # Communication
                'VZ', 'T', 'CMCSA', 'CHTR', 'LBRDK', 'FOXA', 'NWSA',
                # Other
                'BRK-B', 'BRK.A', 'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL',
                'UPS', 'FDX', 'LULU', 'PTON', 'ZM', 'DOCN', 'RPD', 'FROG'
            ]
            
            # Also search by company name using yfinance
            try:
                # Try to get ticker info directly if query looks like a symbol
                if len(query) <= 5 and query.isalpha():
                    ticker_obj = yf.Ticker(query)
                    info = ticker_obj.info
                    if info and info.get('symbol'):
                        results.append({
                            'symbol': info['symbol'],
                            'name': info.get('longName', '') or info.get('shortName', query),
                            'exchange': info.get('exchange', 'NASDAQ'),
                            'sector': info.get('sector', ''),
                            'industry': info.get('industry', ''),
                            'market_cap': info.get('marketCap', 0),
                            'current_price': info.get('currentPrice', 0) or info.get('regularMarketPrice', 0),
                            'change_percent': info.get('regularMarketChangePercent', 0),
                        })
            except Exception:
                pass
            
            # Filter tickers matching query
            matching = []
            for ticker in common_tickers:
                if query in ticker:
                    matching.append(ticker)
                    if len(matching) >= limit:
                        break
            
            # Also try to get company info for exact matches
            for ticker in matching[:limit]:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    
                    # Check if company name matches query
                    company_name = info.get('longName', '') or info.get('shortName', '')
                    if query.lower() in company_name.lower() or query in ticker:
                        results.append({
                            'symbol': ticker,
                            'name': company_name or ticker,
                            'exchange': info.get('exchange', 'NASDAQ'),
                            'sector': info.get('sector', ''),
                            'industry': info.get('industry', ''),
                            'market_cap': info.get('marketCap', 0),
                            'current_price': info.get('currentPrice', 0) or info.get('regularMarketPrice', 0),
                            'change_percent': info.get('regularMarketChangePercent', 0),
                        })
                except Exception as e:
                    logger.debug(f"Error getting info for {ticker}: {e}")
                    # Add basic entry even if info fails
                    if query in ticker:
                        results.append({
                            'symbol': ticker,
                            'name': ticker,
                            'exchange': 'NASDAQ',
                            'sector': '',
                            'industry': '',
                            'market_cap': 0,
                            'current_price': 0,
                            'change_percent': 0,
                        })
            
            # Sort by relevance (exact symbol match first, then name match)
            results.sort(key=lambda x: (
                0 if query == x['symbol'] else 1,
                0 if query.lower() in x['name'].lower() else 1
            ))
            
            self._set_cache(cache_key, results)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching tickers: {e}")
            return []
    
    def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive ticker information."""
        cache_key = f"info_{symbol.upper()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        symbol = symbol.upper().strip()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current quote
            quote = ticker.history(period="1d", interval="1m")
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            if quote is not None and not quote.empty:
                current_price = float(quote['Close'].iloc[-1])
            
            # Get historical data for 52-week high/low
            hist = ticker.history(period="1y")
            week52_high = float(hist['High'].max()) if hist is not None and not hist.empty else current_price
            week52_low = float(hist['Low'].min()) if hist is not None and not hist.empty else current_price
            
            result = {
                'symbol': symbol,
                'name': info.get('longName', '') or info.get('shortName', symbol),
                'exchange': info.get('exchange', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'current_price': current_price,
                'previous_close': info.get('previousClose', current_price),
                'change': current_price - info.get('previousClose', current_price),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0) or info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'eps': info.get('trailingEps', 0),
                'beta': info.get('beta', 0),
                'week52_high': week52_high,
                'week52_low': week52_low,
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'description': info.get('longBusinessSummary', ''),
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ticker info for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote."""
        cache_key = f"quote_{symbol.upper()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        symbol = symbol.upper().strip()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest price
            quote = ticker.history(period="1d", interval="1m")
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            if quote is not None and not quote.empty:
                current_price = float(quote['Close'].iloc[-1])
            
            result = {
                'symbol': symbol,
                'price': current_price,
                'change': current_price - info.get('previousClose', current_price),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0) or info.get('regularMarketVolume', 0),
                'timestamp': datetime.now().isoformat(),
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> List[Dict[str, Any]]:
        """Get historical OHLCV data."""
        cache_key = f"hist_{symbol.upper}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        symbol = symbol.upper().strip()
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist is None or hist.empty:
                return []
            
            # Convert to list of dicts
            result = []
            for idx, row in hist.iterrows():
                try:
                    # Handle different index types
                    if isinstance(idx, pd.Timestamp):
                        timestamp = int(idx.timestamp())
                    elif hasattr(idx, 'timestamp'):
                        timestamp = int(idx.timestamp())
                    else:
                        timestamp = int(pd.Timestamp(idx).timestamp())
                    
                    result.append({
                        'time': timestamp,
                        'open': float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                        'high': float(row['High']) if not pd.isna(row['High']) else 0.0,
                        'low': float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                        'close': float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                        'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                    })
                except Exception as e:
                    logger.debug(f"Error processing row {idx}: {e}")
                    continue
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information."""
        return self.get_ticker_info(symbol)
