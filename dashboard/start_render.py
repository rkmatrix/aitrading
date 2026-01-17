#!/usr/bin/env python3
"""
Universal Render startup script - works regardless of Root Directory setting
Detects environment and imports correctly
"""
import sys
import os
from pathlib import Path

# Get current directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Detect if we're in dashboard/ directory (Root Directory = dashboard/)
# or project root (Root Directory = empty)
is_dashboard_root = not (parent_dir / 'dashboard').exists()

if is_dashboard_root:
    # Root Directory = dashboard/ - files are copied directly
    # Current dir is /opt/render/project/src/ (contains app.py, config.py, etc.)
    sys.path.insert(0, str(current_dir))
    os.chdir(current_dir)
    # Import directly since files are in current directory
    try:
        from app import app, socketio
    except ImportError:
        # Fallback: try dashboard.app if app.py has dashboard imports
        from dashboard.app import app, socketio
else:
    # Root Directory = empty - we're in project root
    # Current dir is /opt/render/project/src/dashboard/
    sys.path.insert(0, str(parent_dir))
    os.chdir(parent_dir)
    from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
