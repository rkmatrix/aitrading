"""
Dashboard Runner - Run from dashboard directory
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
dashboard_dir = Path(__file__).parent
parent_dir = dashboard_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Change to parent directory
os.chdir(parent_dir)

# Now import and run the app
from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Production mode: disable debug and reloader
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
