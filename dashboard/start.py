#!/usr/bin/env python3
"""
Dashboard startup script for Render deployment
Works when Root Directory is set to 'dashboard/'
"""
import sys
import os
from pathlib import Path

# Determine if we're running from dashboard/ directory (Root Directory = dashboard/)
# or from project root (Root Directory = empty)
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Check if dashboard/ directory exists as a subdirectory (Root Directory = empty)
# or if we're already in the copied location (Root Directory = dashboard/)
if (parent_dir / 'dashboard').exists() and (parent_dir / 'dashboard' / 'app.py').exists():
    # Root Directory is empty - we're in project root
    # Add parent directory to path and import from dashboard
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    os.chdir(parent_dir)
    from dashboard.app import app, socketio
else:
    # Root Directory is dashboard/ - files are copied directly
    # Add current directory to path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    os.chdir(current_dir)
    # Import directly (files are in current directory, not in dashboard/ subdirectory)
    from app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Production mode: disable debug and reloader
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
