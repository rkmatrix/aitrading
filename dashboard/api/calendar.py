"""
Calendar API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.database import db
from dashboard.models import Trade
from datetime import datetime, timedelta
from collections import defaultdict

bp = Blueprint('calendar', __name__)


@bp.route('/activity', methods=['GET'])
def get_calendar_activity():
    """Get bot activity by date for calendar."""
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    if not year or not month:
        now = datetime.now()
        year = now.year
        month = now.month
    
    # Get all trades for the month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    trades = db.session.query(Trade).filter(
        Trade.timestamp >= start_date,
        Trade.timestamp < end_date
    ).all()
    
    # Group by date
    daily_activity = defaultdict(lambda: {
        'trades': [],
        'trade_count': 0,
        'total_pnl': 0.0,
        'win_count': 0,
        'loss_count': 0,
    })
    
    for trade in trades:
        trade_date = trade.timestamp.date() if trade.timestamp else datetime.now().date()
        day_key = trade_date.isoformat()
        
        daily_activity[day_key]['trades'].append(trade.to_dict())
        daily_activity[day_key]['trade_count'] += 1
        
        if trade.status == 'FILLED' and trade.extra_data:
            pnl = trade.extra_data.get('pnl', 0)
            daily_activity[day_key]['total_pnl'] += pnl
            if pnl > 0:
                daily_activity[day_key]['win_count'] += 1
            elif pnl < 0:
                daily_activity[day_key]['loss_count'] += 1
    
    # Convert to list format
    result = {}
    for date_str, activity in daily_activity.items():
        result[date_str] = {
            'date': date_str,
            'trade_count': activity['trade_count'],
            'total_pnl': activity['total_pnl'],
            'win_count': activity['win_count'],
            'loss_count': activity['loss_count'],
            'is_profitable': activity['total_pnl'] > 0,
        }
    
    return jsonify({
        'year': year,
        'month': month,
        'activity': result
    })


@bp.route('/day/<date>', methods=['GET'])
def get_day_details(date):
    """Get detailed activity for a specific day."""
    try:
        date_obj = datetime.fromisoformat(date).date()
        start = datetime.combine(date_obj, datetime.min.time())
        end = datetime.combine(date_obj, datetime.max.time())
        
        trades = db.session.query(Trade).filter(
            Trade.timestamp >= start,
            Trade.timestamp <= end
        ).order_by(Trade.timestamp.desc()).all()
        
        total_pnl = sum(
            t.extra_data.get('pnl', 0) 
            for t in trades 
            if t.status == 'FILLED' and t.extra_data
        )
        
        return jsonify({
            'date': date,
            'trades': [t.to_dict() for t in trades],
            'trade_count': len(trades),
            'total_pnl': total_pnl,
            'filled_count': len([t for t in trades if t.status == 'FILLED']),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
