"""
Ticker Analysis API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.database import db
from dashboard.models import Trade, Position
from dashboard.metrics_collector import MetricsCollector
from datetime import datetime, timedelta

bp = Blueprint('ticker_analysis', __name__)
collector = MetricsCollector()


@bp.route('/<symbol>/analysis', methods=['GET'])
def get_ticker_analysis(symbol):
    """Get comprehensive analysis for a ticker."""
    from dashboard.market_data_provider import MarketDataProvider
    
    symbol = symbol.upper()
    market_provider = MarketDataProvider()
    
    # Get market data
    ticker_info = market_provider.get_ticker_info(symbol)
    if not ticker_info:
        return jsonify({"error": "Ticker not found"}), 404
    
    # Get trades for this ticker
    trades = db.session.query(Trade).filter(Trade.symbol == symbol).order_by(Trade.timestamp.desc()).all()
    
    # Calculate performance metrics
    filled_trades = [t for t in trades if t.status == 'FILLED']
    win_count = sum(1 for t in filled_trades if t.extra_data and t.extra_data.get('pnl', 0) > 0)
    win_rate = win_count / len(filled_trades) if filled_trades else 0.0
    
    total_pnl = sum(t.extra_data.get('pnl', 0) for t in filled_trades if t.extra_data) if filled_trades else 0.0
    avg_return = total_pnl / len(filled_trades) if filled_trades else 0.0
    
    # Get best and worst trades
    trade_pnls = [t.extra_data.get('pnl', 0) for t in filled_trades if t.extra_data]
    best_trade = max(trade_pnls) if trade_pnls else 0.0
    worst_trade = min(trade_pnls) if trade_pnls else 0.0
    
    # Get current position
    position = db.session.query(Position).filter(Position.symbol == symbol).first()
    current_metrics = collector.get_current_metrics()
    equity = current_metrics.get('equity', 1)
    
    # Calculate volatility from price history
    hist_data = market_provider.get_historical_data(symbol, period="3mo", interval="1d")
    volatility = 0.0
    if hist_data and len(hist_data) > 1:
        prices = [d['close'] for d in hist_data]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if returns:
            import numpy as np
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    # Get recent signals (from all trades, not just extra_data)
    signals = []
    for trade in trades[:20]:  # Last 20 trades
        signals.append({
            "action": trade.side,
            "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
            "confidence": trade.extra_data.get('confidence', 0.5) if trade.extra_data else 0.5,
            "qty": trade.qty,
            "price": trade.price,
            "status": trade.status,
            "order_type": trade.order_type,
            "filled_qty": trade.filled_qty,
        })
    
    return jsonify({
        "symbol": symbol,
        "overview": ticker_info,
        "performance": {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_trades": len(filled_trades),
            "cumulative_pnl": total_pnl,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        },
        "risk": {
            "position_size": position.qty if position else 0,
            "exposure_pct": (position.market_value / equity) if position and equity > 0 else 0,
            "volatility": volatility,
            "beta": ticker_info.get('beta', 0),
        },
        "trades": [t.to_dict() for t in trades[:50]],
        "signals": signals,
    })


@bp.route('/<symbol>/chart-data', methods=['GET'])
def get_chart_data(symbol):
    """Get chart data for a ticker."""
    from dashboard.market_data_provider import MarketDataProvider
    
    symbol = symbol.upper()
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    
    provider = MarketDataProvider()
    
    # Map period to yfinance periods
    period_map = {
        '1d': ('1d', '5m'),
        '1w': ('5d', '1h'),
        '1m': ('1mo', '1d'),
        '3m': ('3mo', '1d'),
        '6m': ('6mo', '1d'),
        '1y': ('1y', '1d'),
        '5y': ('5y', '1wk'),
    }
    
    if period in period_map:
        p, i = period_map[period]
        bars = provider.get_historical_data(symbol, period=p, interval=i)
    else:
        bars = provider.get_historical_data(symbol, period=period, interval=interval)
    
    return jsonify({
        "candles": bars,
        "symbol": symbol,
        "period": period,
        "interval": interval,
    })
