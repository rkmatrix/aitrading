"""
Main Dashboard Application
"""
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running from dashboard directory
dashboard_dir = Path(__file__).parent
parent_dir = dashboard_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import logging

from dashboard.config import config
from dashboard.database import db, init_db
from dashboard.websocket_handler import WebSocketHandler

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Load configuration
env = os.getenv("FLASK_ENV", "development").lower()
if env not in config:
    env = "development"  # Default to development if invalid env
app.config.from_object(config[env])

# Initialize extensions
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
init_db(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API blueprints
from dashboard.api.bot import bp as bot_bp
from dashboard.api.metrics import bp as metrics_bp
from dashboard.api.trades import bp as trades_bp
from dashboard.api.logs import bp as logs_bp
from dashboard.api.tickers import bp as tickers_bp
from dashboard.api.ticker_analysis import bp as ticker_analysis_bp
from dashboard.api.positions import bp as positions_bp
from dashboard.api.market_data import bp as market_data_bp
from dashboard.api.calendar import bp as calendar_bp

# Register blueprints
app.register_blueprint(bot_bp, url_prefix='/api/bot')
app.register_blueprint(metrics_bp, url_prefix='/api/metrics')
app.register_blueprint(trades_bp, url_prefix='/api/trades')
app.register_blueprint(logs_bp, url_prefix='/api/logs')
app.register_blueprint(tickers_bp, url_prefix='/api/tickers')
app.register_blueprint(ticker_analysis_bp, url_prefix='/api/ticker')
app.register_blueprint(positions_bp, url_prefix='/api/positions')
app.register_blueprint(market_data_bp, url_prefix='/api/market')
app.register_blueprint(calendar_bp, url_prefix='/api/calendar')


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "AITradingBot Dashboard",
        "version": "1.0.0"
    })


# WebSocket handlers are now in websocket_handler.py


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # When running as module, disable reloader to avoid path issues
    socketio.run(app, debug=True, host='0.0.0.0', port=port, use_reloader=False)
