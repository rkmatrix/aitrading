#!/usr/bin/env python3
"""
Dashboard startup script for Render deployment
Works when Root Directory is set to 'dashboard/'
"""
import sys
import os
from pathlib import Path

# Get dashboard directory (where this file is located)
dashboard_dir = Path(__file__).resolve().parent
parent_dir = dashboard_dir.parent

# Add parent directory to Python path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Change to parent directory (project root)
os.chdir(parent_dir)

# Import and run the dashboard app
from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Production mode: disable debug and reloader
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
