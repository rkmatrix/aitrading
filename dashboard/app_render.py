#!/usr/bin/env python3
"""
Render-compatible app wrapper - ALWAYS use Root Directory = empty approach
This script ensures we're in project root and imports work correctly
"""
import sys
import os
from pathlib import Path

# Get current directory (where this file is)
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# ALWAYS go to project root and use dashboard.app import
# This works regardless of Root Directory setting because:
# - If Root Directory = empty: we're already in project root
# - If Root Directory = dashboard/: we go up one level to project root

# Add parent directory to path (project root)
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Change to project root
os.chdir(parent_dir)

# Import from dashboard - this ALWAYS works when we're in project root
from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    print(f"Working directory: {os.getcwd()}")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
