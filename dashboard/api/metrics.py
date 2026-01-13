"""
Metrics API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.metrics_collector import MetricsCollector
from dashboard.database import db
from dashboard.models import Metric as MetricModel
from datetime import datetime, timedelta

bp = Blueprint('metrics', __name__)
collector = MetricsCollector()


@bp.route('/current', methods=['GET'])
def get_current_metrics():
    """Get current metrics."""
    metrics = collector.get_current_metrics()
    
    # Calculate additional metrics
    try:
        # Get recent metrics for daily P&L calculation
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_metric = db.session.query(MetricModel).filter(
            MetricModel.timestamp >= yesterday
        ).order_by(MetricModel.timestamp.asc()).first()
        
        if yesterday_metric:
            metrics['daily_pnl'] = metrics['total_pnl'] - yesterday_metric.total_pnl
        
        # Calculate win rate from trades
        from dashboard.models import Trade
        filled_trades = db.session.query(Trade).filter(Trade.status == 'FILLED').all()
        if filled_trades:
            winning_trades = sum(1 for t in filled_trades if t.extra_data and t.extra_data.get('pnl', 0) > 0)
            metrics['win_rate'] = winning_trades / len(filled_trades) if filled_trades else 0.0
        else:
            metrics['win_rate'] = 0.0
        
        # Calculate max drawdown
        all_metrics = db.session.query(MetricModel).order_by(MetricModel.timestamp.desc()).limit(100).all()
        if all_metrics and len(all_metrics) > 1:
            equities = [m.equity for m in reversed(all_metrics)]
            peak = equities[0]
            max_dd = 0.0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            metrics['max_drawdown'] = max_dd
        else:
            metrics['max_drawdown'] = 0.0
        
        # Save to database
        metric_record = MetricModel(
            timestamp=datetime.now(),
            equity=metrics['equity'],
            buying_power=metrics['buying_power'],
            cash=metrics['cash'],
            portfolio_value=metrics['portfolio_value'],
            realized_pnl=metrics['realized_pnl'],
            unrealized_pnl=metrics['unrealized_pnl'],
            total_pnl=metrics['total_pnl'],
            daily_pnl=metrics.get('daily_pnl', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
        )
        db.session.add(metric_record)
        db.session.commit()
    except Exception as e:
        print(f"Failed to save metrics: {e}")
    
    return jsonify(metrics)


@bp.route('/history', methods=['GET'])
def get_metrics_history():
    """Get historical metrics."""
    hours = request.args.get('hours', 24, type=int)
    since = datetime.now() - timedelta(hours=hours)
    
    try:
        metrics = db.session.query(MetricModel).filter(
            MetricModel.timestamp >= since
        ).order_by(MetricModel.timestamp.desc()).all()
        
        return jsonify({
            "metrics": [m.to_dict() for m in metrics]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
