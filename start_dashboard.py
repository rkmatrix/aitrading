#!/usr/bin/env python3
"""
Dashboard startup script for Render deployment
Works from project root regardless of Root Directory setting
"""
import sys
import os
from pathlib import Path

# Get project root (where this file is located)
project_root = Path(__file__).resolve().parent

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root
os.chdir(project_root)

# Import and run the dashboard app
from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Production mode: disable debug and reloader
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
