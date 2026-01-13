"""
Database Models for Dashboard
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    """Trade execution record."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(100), unique=True, nullable=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    filled_qty = Column(Float, default=0.0)
    filled_avg_price = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, index=True)  # NEW, FILLED, REJECTED, CANCELED
    order_type = Column(String(20), default="MARKET")
    limit_price = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    filled_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata (SQLAlchemy reserved)
    
    def to_dict(self):
        return {
            "id": self.id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "status": self.status,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "error_message": self.error_message,
            "metadata": self.extra_data,  # Keep API name as metadata for compatibility
        }


class Position(Base):
    """Current position snapshot."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    qty = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    market_value = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Metric(Base):
    """Portfolio metrics snapshot."""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equity = Column(Float, nullable=False)
    buying_power = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    portfolio_value = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata (SQLAlchemy reserved)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "metadata": self.extra_data,  # Keep API name as metadata for compatibility
        }


class LogEntry(Base):
    """Log entry for dashboard display."""
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    component = Column(String(100), nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata (SQLAlchemy reserved)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "message": self.message,
            "component": self.component,
            "metadata": self.extra_data,  # Keep API name as metadata for compatibility
        }


class TickerConfig(Base):
    """Ticker configuration for trading."""
    __tablename__ = "ticker_configs"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    enabled = Column(Boolean, default=True, index=True)
    halted = Column(Boolean, default=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    removed_at = Column(DateTime, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata (SQLAlchemy reserved)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "enabled": self.enabled,
            "halted": self.halted,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "removed_at": self.removed_at.isoformat() if self.removed_at else None,
            "metadata": self.extra_data,  # Keep API name as metadata for compatibility
        }
