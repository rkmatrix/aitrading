"""
Unified Broker Adapter Interface
---------------------------------
Provides a unified interface for different broker implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Standardized order request."""
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"  # "DAY", "GTC", "IOC", "FOK"
    client_order_id: Optional[str] = None
    tag: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class OrderResponse:
    """Standardized order response."""
    order_id: str
    symbol: str
    side: str
    qty: float
    status: str  # "NEW", "FILLED", "PARTIALLY_FILLED", "CANCELED", "REJECTED"
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    """Standardized position representation."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class AccountInfo:
    """Standardized account information."""
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    day_trading_buying_power: Optional[float] = None
    pattern_day_trader: bool = False
    account_blocked: bool = False
    trading_blocked: bool = False


class BrokerAdapter(ABC):
    """
    Unified broker adapter interface.
    
    All broker implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, mode: str = "PAPER"):
        """
        Initialize broker adapter.
        
        Args:
            mode: Trading mode ("PAPER", "LIVE", "DEMO")
        """
        self.mode = mode.upper()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account information."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        pass
    
    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit an order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last traded price for symbol."""
        pass
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Default implementation uses MarketClock.
        Can be overridden by broker-specific implementations.
        """
        from ai.market.market_clock import MarketClock
        clock = MarketClock()
        return clock.is_open()
    
    def validate_order(self, order: OrderRequest) -> tuple[bool, str]:
        """
        Validate order before submission.
        
        Can be overridden for broker-specific validation.
        """
        # Basic validation
        if not order.symbol:
            return False, "Symbol is required"
        
        if order.qty <= 0:
            return False, "Quantity must be positive"
        
        if order.side.upper() not in ("BUY", "SELL"):
            return False, f"Invalid side: {order.side}"
        
        if order.order_type.upper() == "LIMIT" and order.limit_price is None:
            return False, "Limit price required for limit orders"
        
        return True, "OK"
    
    def get_name(self) -> str:
        """Get broker name."""
        return self.__class__.__name__


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca broker adapter implementation.
    
    Wraps AlpacaClient to provide unified interface.
    """
    
    def __init__(self, mode: str = "PAPER"):
        super().__init__(mode)
        
        try:
            from ai.execution.broker_alpaca_live import AlpacaClient
            self.client = AlpacaClient()
        except Exception as e:
            self.logger.error("Failed to initialize AlpacaClient: %s", e)
            raise
    
    def get_account(self) -> AccountInfo:
        """Get account information."""
        try:
            acct_dict = self.client.get_account()
            
            return AccountInfo(
                equity=float(acct_dict.get("equity", 0.0)),
                buying_power=float(acct_dict.get("buying_power", 0.0)),
                cash=float(acct_dict.get("cash", acct_dict.get("buying_power", 0.0))),
                portfolio_value=float(acct_dict.get("equity", 0.0)),
            )
        except Exception as e:
            self.logger.error("Failed to get account: %s", e)
            raise
    
    def get_positions(self) -> List[Position]:
        """Get all positions."""
        try:
            positions_data = self.client.get_positions()
            positions = []
            
            for p in positions_data:
                positions.append(Position(
                    symbol=str(p.get("symbol", "")),
                    qty=float(p.get("qty", 0.0)),
                    avg_entry_price=float(p.get("price", 0.0)),
                ))
            
            return positions
        except Exception as e:
            self.logger.error("Failed to get positions: %s", e)
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol.upper():
                return p
        return None
    
    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit order."""
        # Validate first
        is_valid, error = self.validate_order(order)
        if not is_valid:
            return OrderResponse(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                status="REJECTED",
                error=error,
            )
        
        try:
            order_dict = {
                "symbol": order.symbol,
                "side": order.side.lower(),
                "qty": order.qty,
                "order_type": order.order_type.lower(),
                "limit_price": order.limit_price,
                "client_order_id": order.client_order_id,
                "tag": order.tag,
                "meta": order.meta or {},
            }
            
            resp = self.client.place_order(order_dict)
            
            # Convert response to OrderResponse
            if isinstance(resp, dict) and "error" in resp:
                return OrderResponse(
                    order_id="",
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    status="REJECTED",
                    error=resp.get("error", "Unknown error"),
                )
            
            # Extract order ID from response
            order_id = str(getattr(resp, "id", "")) if hasattr(resp, "id") else ""
            
            return OrderResponse(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                status="NEW",
                raw_response={"resp": str(resp)},
            )
        except Exception as e:
            self.logger.error("Failed to submit order: %s", e, exc_info=True)
            return OrderResponse(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                status="REJECTED",
                error=str(e),
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            # AlpacaClient doesn't expose cancel directly, would need to add
            self.logger.warning("Cancel order not implemented for AlpacaClient")
            return False
        except Exception as e:
            self.logger.error("Failed to cancel order: %s", e)
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status."""
        try:
            # Would need to implement using AlpacaClient
            self.logger.warning("Get order status not fully implemented")
            return None
        except Exception as e:
            self.logger.error("Failed to get order status: %s", e)
            return None
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price."""
        try:
            return self.client.get_last_price(symbol)
        except Exception as e:
            self.logger.error("Failed to get last price for %s: %s", symbol, e)
            return None


class DummyBrokerAdapter(BrokerAdapter):
    """
    Dummy broker adapter for testing/demo.
    
    Does not execute real trades.
    """
    
    def __init__(self, mode: str = "DEMO"):
        super().__init__(mode)
        self._orders: Dict[str, OrderResponse] = {}
        self._positions: Dict[str, Position] = {}
        self._account = AccountInfo(
            equity=100000.0,
            buying_power=200000.0,
            cash=100000.0,
            portfolio_value=100000.0,
        )
    
    def get_account(self) -> AccountInfo:
        return self._account
    
    def get_positions(self) -> List[Position]:
        return list(self._positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol.upper())
    
    def submit_order(self, order: OrderRequest) -> OrderResponse:
        self.logger.info("DUMMY: Submitting order %s %s %s", order.side, order.qty, order.symbol)
        
        order_id = f"DUMMY_{len(self._orders)}"
        response = OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            status="FILLED",
            filled_qty=order.qty,
            filled_avg_price=100.0,  # Dummy price
        )
        
        self._orders[order_id] = response
        
        # Update position
        current = self._positions.get(order.symbol.upper())
        if current:
            new_qty = current.qty + (order.qty if order.side == "BUY" else -order.qty)
            if new_qty != 0:
                self._positions[order.symbol.upper()] = Position(
                    symbol=order.symbol.upper(),
                    qty=new_qty,
                    avg_entry_price=100.0,
                )
            else:
                self._positions.pop(order.symbol.upper(), None)
        else:
            if order.side == "BUY":
                self._positions[order.symbol.upper()] = Position(
                    symbol=order.symbol.upper(),
                    qty=order.qty,
                    avg_entry_price=100.0,
                )
        
        return response
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "CANCELED"
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        return self._orders.get(order_id)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        return 100.0  # Dummy price


def create_broker_adapter(broker_type: str = "alpaca", mode: str = "PAPER") -> BrokerAdapter:
    """
    Factory function to create broker adapter.
    
    Args:
        broker_type: Type of broker ("alpaca", "dummy")
        mode: Trading mode ("PAPER", "LIVE", "DEMO")
    
    Returns:
        BrokerAdapter instance
    """
    broker_type = broker_type.lower()
    
    if broker_type == "alpaca":
        return AlpacaBrokerAdapter(mode=mode)
    elif broker_type == "dummy":
        return DummyBrokerAdapter(mode=mode)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
