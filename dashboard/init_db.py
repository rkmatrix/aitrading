"""Initialize dashboard database"""
import sys
from pathlib import Path

# Add parent directory to path
dashboard_dir = Path(__file__).parent
parent_dir = dashboard_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from dashboard.app import app
from dashboard.database import db
from dashboard.models import Trade, Position, Metric, LogEntry, TickerConfig

with app.app_context():
    db.create_all()
    print("Database tables created successfully!")
