"""
Tests for ConfigValidator
"""

import unittest
import os
from ai.utils.config_validator import ConfigValidator


class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    def test_required_field_present(self):
        """Test validation with required field present."""
        self.validator.add_rule("TEST_KEY", required=True)
        config = {"TEST_KEY": "test_value"}
        
        is_valid, errors = self.validator.validate(config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_required_field_missing(self):
        """Test validation with required field missing."""
        self.validator.add_rule("TEST_KEY", required=True)
        config = {}
        
        is_valid, errors = self.validator.validate(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_optional_field_with_default(self):
        """Test optional field with default value."""
        self.validator.add_rule("OPTIONAL_KEY", required=False, default="default_value")
        config = {}
        
        is_valid, errors = self.validator.validate(config)
        self.assertTrue(is_valid)
        self.assertEqual(config.get("OPTIONAL_KEY"), "default_value")
    
    def test_validator_function(self):
        """Test custom validator function."""
        self.validator.add_rule(
            "MODE",
            required=True,
            validator=lambda x: x.upper() in ["PAPER", "LIVE"],
        )
        
        # Valid value
        config = {"MODE": "PAPER"}
        is_valid, errors = self.validator.validate(config)
        self.assertTrue(is_valid)
        
        # Invalid value
        config = {"MODE": "INVALID"}
        is_valid, errors = self.validator.validate(config)
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
