"""
Trades API Endpoints
"""
from flask import Blueprint, jsonify, request, send_file
from dashboard.database import db
from dashboard.models import Trade
from datetime import datetime
import csv
import io

bp = Blueprint('trades', __name__)


@bp.route('', methods=['GET'])
def get_trades():
    """Get trades with optional filters."""
    symbol = request.args.get('symbol')
    status = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)
    
    query = db.session.query(Trade)
    
    if symbol:
        query = query.filter(Trade.symbol == symbol.upper())
    if status:
        query = query.filter(Trade.status == status.upper())
    
    trades = query.order_by(Trade.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        "trades": [t.to_dict() for t in trades],
        "count": len(trades)
    })


@bp.route('/<int:trade_id>', methods=['GET'])
def get_trade(trade_id):
    """Get specific trade."""
    trade = db.session.get(Trade, trade_id)
    if not trade:
        from flask import abort
        abort(404)
    return jsonify(trade.to_dict())


@bp.route('/export', methods=['GET'])
def export_trades():
    """Export trades to CSV."""
    trades = db.session.query(Trade).order_by(Trade.timestamp.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'ID', 'Order ID', 'Symbol', 'Side', 'Quantity', 'Price',
        'Filled Qty', 'Filled Avg Price', 'Status', 'Timestamp'
    ])
    
    # Write data
    for trade in trades:
        writer.writerow([
            trade.id,
            trade.order_id,
            trade.symbol,
            trade.side,
            trade.qty,
            trade.price,
            trade.filled_qty,
            trade.filled_avg_price,
            trade.status,
            trade.timestamp.isoformat() if trade.timestamp else '',
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'trades_{datetime.now().strftime("%Y%m%d")}.csv'
    )
