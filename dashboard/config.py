"""
Dashboard Configuration
"""
import os
from pathlib import Path

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Path('dashboard/data/dashboard.db').absolute()}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Bot integration
    BOT_LOG_FILE = Path("data/logs/phase26_realtime_live.log")
    BOT_STATE_FILE = Path("data/runtime/phase26_bot_state.json")
    KILL_SWITCH_FILE = Path("data/runtime/trading_disabled.flag")
    
    # WebSocket
    WEBSOCKET_PING_INTERVAL = 25
    WEBSOCKET_PING_TIMEOUT = 10
    
    # Update intervals (seconds)
    METRICS_UPDATE_INTERVAL = 1.0
    LOGS_UPDATE_INTERVAL = 0.5
    TRADES_UPDATE_INTERVAL = 1.0
    
    # Pagination
    TRADES_PER_PAGE = 50
    LOGS_PER_PAGE = 100

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY")
    if not SECRET_KEY:
        # Use a default in development, but warn in production
        import warnings
        warnings.warn("DASHBOARD_SECRET_KEY not set, using default. Set this in production!")
        SECRET_KEY = "dev-secret-key-change-in-production"

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
