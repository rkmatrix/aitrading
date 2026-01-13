"""
Configuration Validator
-----------------------
Validates configuration and environment variables on startup.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfigRule:
    """Rule for validating a configuration value."""
    key: str
    required: bool = True
    default: Any = None
    validator: Optional[callable] = None
    error_message: Optional[str] = None


class ConfigValidator:
    """
    Validates configuration and environment variables.
    
    Usage:
        validator = ConfigValidator()
        validator.add_rule("MODE", required=True, validator=lambda x: x in ["PAPER", "LIVE"])
        is_valid, errors = validator.validate()
    """
    
    def __init__(self):
        self.rules: List[ConfigRule] = []
        self.logger = logging.getLogger("ConfigValidator")
    
    def add_rule(
        self,
        key: str,
        required: bool = True,
        default: Any = None,
        validator: Optional[callable] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Add a validation rule."""
        rule = ConfigRule(
            key=key,
            required=required,
            default=default,
            validator=validator,
            error_message=error_message,
        )
        self.rules.append(rule)
    
    def validate(self, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Validate all rules.
        
        Args:
            config: Optional config dict. If None, reads from environment variables.
        
        Returns:
            Tuple of (is_valid: bool, errors: list[str])
        """
        errors = []
        
        if config is None:
            config = dict(os.environ)
        
        for rule in self.rules:
            value = config.get(rule.key)
            
            # Check if required
            if rule.required and value is None:
                error_msg = rule.error_message or f"Required configuration '{rule.key}' is missing"
                errors.append(error_msg)
                self.logger.error("❌ %s", error_msg)
                continue
            
            # Use default if not provided
            if value is None and rule.default is not None:
                value = rule.default
                config[rule.key] = value
                self.logger.info("Using default for %s: %s", rule.key, rule.default)
            
            # Validate value if validator provided
            if value is not None and rule.validator is not None:
                try:
                    if not rule.validator(value):
                        error_msg = (
                            rule.error_message or
                            f"Invalid value for '{rule.key}': {value}"
                        )
                        errors.append(error_msg)
                        self.logger.error("❌ %s", error_msg)
                except Exception as e:
                    error_msg = (
                        rule.error_message or
                        f"Validation error for '{rule.key}': {e}"
                    )
                    errors.append(error_msg)
                    self.logger.error("❌ %s", error_msg)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info("✅ Configuration validation passed")
        else:
            self.logger.error("❌ Configuration validation failed with %d errors", len(errors))
        
        return is_valid, errors
    
    def validate_trading_config(self) -> Tuple[bool, List[str]]:
        """
        Validate trading-specific configuration.
        
        Returns:
            Tuple of (is_valid: bool, errors: list[str])
        """
        # Mode validation
        self.add_rule(
            "MODE",
            required=False,
            default="PAPER",
            validator=lambda x: x.upper() in ["PAPER", "LIVE", "DEMO"],
            error_message="MODE must be PAPER, LIVE, or DEMO",
        )
        
        # API keys (validated separately, but check presence)
        self.add_rule(
            "APCA_API_KEY_ID",
            required=True,
            error_message="APCA_API_KEY_ID is required for trading",
        )
        
        self.add_rule(
            "APCA_API_SECRET_KEY",
            required=True,
            error_message="APCA_API_SECRET_KEY is required for trading",
        )
        
        # Base URL (optional, will use defaults)
        self.add_rule(
            "APCA_API_BASE_URL",
            required=False,
            validator=lambda x: x.startswith("http"),
            error_message="APCA_API_BASE_URL must be a valid URL",
        )
        
        return self.validate()


def validate_startup_config() -> bool:
    """
    Validate configuration on startup.
    
    Returns:
        True if valid, raises RuntimeError if invalid
    """
    validator = ConfigValidator()
    is_valid, errors = validator.validate_trading_config()
    
    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    return True
