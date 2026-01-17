#!/usr/bin/env python3
"""
Dashboard startup script for Render deployment
Works when Root Directory is set to 'dashboard/'
When Root Directory is 'dashboard/', Render copies files directly without 'dashboard/' prefix
"""
import sys
import os
from pathlib import Path

# Get current directory (where this file is located)
current_dir = Path(__file__).resolve().parent

# Add current directory to Python path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Change to current directory
os.chdir(current_dir)

# Import and run the dashboard app
# When Root Directory is 'dashboard/', files are copied directly, so import from 'app' not 'dashboard.app'
from app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Production mode: disable debug and reloader
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
