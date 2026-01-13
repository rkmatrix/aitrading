"""
Tickers API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.ticker_manager import TickerManager

bp = Blueprint('tickers', __name__)
manager = TickerManager()


@bp.route('', methods=['GET'])
def get_tickers():
    """Get all trading tickers."""
    tickers = manager.get_tickers()
    return jsonify({"tickers": tickers})


@bp.route('/add', methods=['POST'])
def add_ticker():
    """Add ticker to trading list."""
    data = request.get_json() or {}
    symbol = data.get('symbol', '').strip()
    
    if not symbol:
        return jsonify({"success": False, "message": "Symbol is required"}), 400
    
    result = manager.add_ticker(symbol)
    return jsonify(result)


@bp.route('/remove', methods=['POST'])
def remove_ticker():
    """Remove ticker from trading list."""
    data = request.get_json() or {}
    symbol = data.get('symbol', '').strip()
    
    if not symbol:
        return jsonify({"success": False, "message": "Symbol is required"}), 400
    
    result = manager.remove_ticker(symbol)
    return jsonify(result)


@bp.route('/halt', methods=['POST'])
def halt_ticker():
    """Halt trading for a ticker."""
    data = request.get_json() or {}
    symbol = data.get('symbol', '').strip()
    
    if not symbol:
        return jsonify({"success": False, "message": "Symbol is required"}), 400
    
    result = manager.halt_ticker(symbol)
    return jsonify(result)


@bp.route('/resume', methods=['POST'])
def resume_ticker():
    """Resume trading for a ticker."""
    data = request.get_json() or {}
    symbol = data.get('symbol', '').strip()
    
    if not symbol:
        return jsonify({"success": False, "message": "Symbol is required"}), 400
    
    result = manager.resume_ticker(symbol)
    return jsonify(result)


@bp.route('/search', methods=['GET'])
def search_tickers():
    """Search for tickers (deprecated - use /api/market/search)."""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({"results": []})
    
    results = manager.search_tickers(query)
    return jsonify({"results": results})
