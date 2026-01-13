"""
Tests for OrderValidator
"""

import unittest
from ai.execution.order_validator import OrderValidator, ValidationConfig


class TestOrderValidator(unittest.TestCase):
    """Test cases for OrderValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(
            min_qty=1.0,
            max_qty=1000.0,
            min_notional=100.0,
        )
        self.validator = OrderValidator(config=self.config)
    
    def test_valid_buy_order(self):
        """Test valid buy order."""
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 10,
            "order_type": "market",
        }
        account = {"buying_power": 10000.0}
        
        is_valid, error = self.validator.validate(order, account, last_price=150.0)
        self.assertTrue(is_valid, f"Order should be valid: {error}")
    
    def test_invalid_symbol(self):
        """Test invalid symbol."""
        order = {
            "symbol": "INVALID_SYMBOL_TOO_LONG",
            "side": "BUY",
            "qty": 10,
            "order_type": "market",
        }
        
        is_valid, error = self.validator.validate(order)
        self.assertFalse(is_valid)
        self.assertIn("symbol", error.lower())
    
    def test_invalid_quantity(self):
        """Test invalid quantity."""
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 0.5,  # Below min_qty
            "order_type": "market",
        }
        
        is_valid, error = self.validator.validate(order)
        self.assertFalse(is_valid)
        self.assertIn("quantity", error.lower())
    
    def test_insufficient_buying_power(self):
        """Test insufficient buying power."""
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 100,
            "order_type": "market",
        }
        account = {"buying_power": 1000.0}  # Not enough for 100 shares at $150
        
        is_valid, error = self.validator.validate(order, account, last_price=150.0)
        self.assertFalse(is_valid)
        self.assertIn("buying power", error.lower())
    
    def test_invalid_price(self):
        """Test invalid price."""
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 10,
            "order_type": "market",
        }
        
        is_valid, error = self.validator.validate(order, last_price=-10.0)
        self.assertFalse(is_valid)
        self.assertIn("price", error.lower())
    
    def test_limit_order_without_price(self):
        """Test limit order without limit_price."""
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 10,
            "order_type": "limit",
        }
        
        is_valid, error = self.validator.validate(order)
        self.assertFalse(is_valid)
        self.assertIn("limit", error.lower())


if __name__ == "__main__":
    unittest.main()
