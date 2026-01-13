"""
WebSocket Handler for Real-time Updates
"""
import logging
from flask_socketio import emit
from dashboard.database import db
from dashboard.models import Trade, Metric, LogEntry

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections and broadcasts."""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            logger.info("WebSocket client connected")
            emit('connected', {'message': 'Connected to dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("WebSocket client disconnected")
    
    def broadcast_bot_status(self, status_data):
        """Broadcast bot status update."""
        self.socketio.emit('bot.status.update', status_data)
    
    def broadcast_trade(self, trade_data):
        """Broadcast new trade."""
        self.socketio.emit('trade.executed', {'trade': trade_data})
    
    def broadcast_metric(self, metric_data):
        """Broadcast metric update."""
        self.socketio.emit('metric.update', {'metrics': metric_data})
    
    def broadcast_log(self, log_data):
        """Broadcast log entry."""
        self.socketio.emit('log.entry', {'log': log_data})
    
    def broadcast_ticker_status(self, symbol, status):
        """Broadcast ticker status update."""
        self.socketio.emit('ticker.status.update', {
            'symbol': symbol,
            'status': status
        })
    
    def broadcast_error(self, error_data):
        """Broadcast error."""
        self.socketio.emit('error.occurred', error_data)
