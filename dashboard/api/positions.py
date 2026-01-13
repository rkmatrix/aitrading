"""
Positions API Endpoints
"""
from flask import Blueprint, jsonify
from dashboard.metrics_collector import MetricsCollector

bp = Blueprint('positions', __name__)
collector = MetricsCollector()


@bp.route('', methods=['GET'])
def get_positions():
    """Get current positions."""
    positions = collector.get_positions()
    return jsonify({"positions": positions})
