"""
Test dashboard startup to check for errors
"""
import sys
from pathlib import Path

# Add parent directory to path
dashboard_dir = Path(__file__).parent
parent_dir = dashboard_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    print("Testing dashboard imports...")
    from dashboard.config import config
    print("OK: Config imported")
    
    from dashboard.models import Trade, Position, Metric, LogEntry, TickerConfig
    print("OK: Models imported")
    
    from dashboard.database import db
    print("OK: Database imported")
    
    from dashboard.app import app, socketio
    print("OK: App imported")
    
    print("\nALL GOOD: All imports successful! Dashboard should start without errors.")
    print("\nTo start the dashboard, run:")
    print("  python -m dashboard.app")
    print("  or")
    print("  cd dashboard && python app.py")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
