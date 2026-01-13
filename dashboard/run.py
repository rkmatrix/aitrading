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
    # Disable reloader to avoid path issues when running from dashboard directory
    socketio.run(app, debug=True, host='0.0.0.0', port=port, use_reloader=False)
