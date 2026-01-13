"""
Database initialization and utilities
"""
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path

db = SQLAlchemy()

def init_db(app):
    """Initialize database."""
    db.init_app(app)
    
    # Create data directory
    data_dir = Path("dashboard/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tables
    with app.app_context():
        db.create_all()
        print("Database initialized")
