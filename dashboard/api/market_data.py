"""
Market Data API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.market_data_provider import MarketDataProvider

bp = Blueprint('market_data', __name__)
provider = MarketDataProvider()


@bp.route('/search', methods=['GET'])
def search_tickers():
    """Search for tickers."""
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 20, type=int)
    
    if not query or len(query) < 1:
        return jsonify({"results": []})
    
    results = provider.search_tickers(query, limit=limit)
    return jsonify({"results": results})


@bp.route('/ticker/<symbol>/info', methods=['GET'])
def get_ticker_info(symbol):
    """Get ticker information."""
    info = provider.get_ticker_info(symbol)
    if info:
        return jsonify(info)
    return jsonify({"error": "Ticker not found"}), 404


@bp.route('/ticker/<symbol>/quote', methods=['GET'])
def get_quote(symbol):
    """Get real-time quote."""
    quote = provider.get_quote(symbol)
    if quote:
        return jsonify(quote)
    return jsonify({"error": "Quote not available"}), 404


@bp.route('/ticker/<symbol>/history', methods=['GET'])
def get_history(symbol):
    """Get historical data."""
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    
    data = provider.get_historical_data(symbol, period=period, interval=interval)
    return jsonify({"data": data, "symbol": symbol})
