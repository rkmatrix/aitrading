"""
Logs API Endpoints
"""
from flask import Blueprint, jsonify, request
from pathlib import Path
import re
from datetime import datetime

bp = Blueprint('logs', __name__)


@bp.route('/stream', methods=['GET'])
def stream_logs():
    """Get recent log entries."""
    log_file = Path("data/logs/phase26_realtime_live.log")
    limit = request.args.get('limit', 100, type=int)
    level = request.args.get('level')  # INFO, WARNING, ERROR
    
    if not log_file.exists():
        return jsonify({"logs": []})
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse log entries
        logs = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            
            # Parse log format: 2026-01-10 07:20:32,246 | INFO | Phase26RealtimeUltra | Message
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| (\w+) \| ([^|]+) \| (.+)', line)
            if match:
                timestamp_str, log_level, component, message = match.groups()
                
                if level and log_level != level:
                    continue
                
                logs.append({
                    "timestamp": timestamp_str,
                    "level": log_level,
                    "component": component.strip(),
                    "message": message.strip(),
                })
        
        return jsonify({"logs": logs[-limit:]})
    except Exception as e:
        return jsonify({"error": str(e), "logs": []}), 500
