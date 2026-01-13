"""
Centralized Risk Manager
------------------------
Consolidates risk checking logic from multiple sources into a unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    severity: str = "info"  # "info", "warning", "error"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Centralized risk management system.
    
    Consolidates risk checks from:
    - RiskEnvelopeController
    - SafetyGuard
    - OrderValidator
    - Position limits
    - Drawdown limits
    """
    
    def __init__(
        self,
        risk_envelope: Optional[Any] = None,
        safety_guard: Optional[Any] = None,
        order_validator: Optional[Any] = None,
    ):
        """
        Initialize risk manager.
        
        Args:
            risk_envelope: RiskEnvelopeController instance (optional)
            safety_guard: SafetyGuard instance (optional)
            order_validator: OrderValidator instance (optional)
        """
        self.risk_envelope = risk_envelope
        self.safety_guard = safety_guard
        self.order_validator = order_validator
        self.logger = logging.getLogger("RiskManager")
        
        # Risk check history
        self._check_history: List[Dict[str, Any]] = []
    
    def check_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        account: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskCheckResult:
        """
        Comprehensive risk check for an order.
        
        Args:
            symbol: Symbol to trade
            side: "BUY" or "SELL"
            qty: Quantity
            price: Price
            account: Account information
            portfolio: Portfolio information
            context: Additional context
        
        Returns:
            RiskCheckResult with allowed status and reason
        """
        context = context or {}
        checks: List[Tuple[bool, str, str]] = []
        
        # 1. Order validation
        if self.order_validator:
            try:
                order_dict = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "order_type": "MARKET",
                    "price": price,
                }
                is_valid, error = self.order_validator.validate(order_dict, account, price)
                checks.append((is_valid, f"Order validation: {error}", "error" if not is_valid else "info"))
                if not is_valid:
                    return RiskCheckResult(
                        allowed=False,
                        reason=f"Order validation failed: {error}",
                        severity="error",
                        metadata={"check": "order_validation"},
                    )
            except Exception as e:
                self.logger.warning("Order validation check failed: %s", e)
        
        # 2. Safety guard check
        if self.safety_guard:
            try:
                # Extract context for safety guard
                equity = portfolio.get("equity", 0.0) if portfolio else 0.0
                current_qty = context.get("current_qty", 0.0)
                volatility = context.get("volatility", 0.2)
                max_drawdown = portfolio.get("max_drawdown", 0.0) if portfolio else 0.0
                confidence = context.get("confidence", 0.5)
                conflict = context.get("conflict", 0.0)
                
                # Note: SafetyGuard API may vary, adjust as needed
                if hasattr(self.safety_guard, "allow_trade"):
                    allowed, reason = self.safety_guard.allow_trade(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        price=price,
                        confidence=confidence,
                        conflict=conflict,
                        volatility=volatility,
                        current_qty=current_qty,
                        equity=equity,
                        max_drawdown=max_drawdown,
                    )
                    checks.append((allowed, f"Safety guard: {reason}", "warning" if not allowed else "info"))
                    if not allowed:
                        return RiskCheckResult(
                            allowed=False,
                            reason=f"Safety guard blocked: {reason}",
                            severity="warning",
                            metadata={"check": "safety_guard"},
                        )
            except Exception as e:
                self.logger.warning("Safety guard check failed: %s", e)
        
        # 3. Risk envelope check
        if self.risk_envelope and portfolio:
            try:
                # Build order dict for risk envelope
                order_dict = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                }
                
                # Risk envelope evaluate method
                if hasattr(self.risk_envelope, "evaluate"):
                    adjusted_order, meta = self.risk_envelope.evaluate(portfolio, order_dict)
                    
                    if meta.get("hard_kill", False):
                        return RiskCheckResult(
                            allowed=False,
                            reason=f"Risk envelope hard kill: {meta.get('reason', 'unknown')}",
                            severity="error",
                            metadata={"check": "risk_envelope", "meta": meta},
                        )
                    
                    if meta.get("clamped", False):
                        return RiskCheckResult(
                            allowed=True,
                            reason=f"Risk envelope clamped order: {meta.get('reason', 'unknown')}",
                            severity="warning",
                            metadata={"check": "risk_envelope", "adjusted_order": adjusted_order, "meta": meta},
                        )
            except Exception as e:
                self.logger.warning("Risk envelope check failed: %s", e)
        
        # 4. Basic sanity checks
        sanity_checks = self._basic_sanity_checks(symbol, side, qty, price, account, portfolio)
        if not sanity_checks[0]:
            return RiskCheckResult(
                allowed=False,
                reason=sanity_checks[1],
                severity="error",
                metadata={"check": "sanity"},
            )
        
        # All checks passed
        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            severity="info",
            metadata={"checks_performed": len(checks)},
        )
    
    def _basic_sanity_checks(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        account: Optional[Dict[str, Any]],
        portfolio: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Basic sanity checks."""
        # Check symbol
        if not symbol or len(symbol) > 10:
            return False, "Invalid symbol"
        
        # Check side
        if side.upper() not in ("BUY", "SELL"):
            return False, f"Invalid side: {side}"
        
        # Check quantity
        if qty <= 0:
            return False, "Quantity must be positive"
        
        # Check price
        if price <= 0:
            return False, "Price must be positive"
        
        # Check buying power for buy orders
        if side.upper() == "BUY" and account:
            buying_power = account.get("buying_power", 0.0)
            required = qty * price
            if required > buying_power * 0.95:  # Use max 95% of buying power
                return False, f"Insufficient buying power: required ${required:.2f}, available ${buying_power:.2f}"
        
        return True, "OK"
    
    def get_check_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent risk check history."""
        return self._check_history[-limit:]
    
    def log_check(self, result: RiskCheckResult, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a risk check result."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "allowed": result.allowed,
            "reason": result.reason,
            "severity": result.severity,
            "metadata": result.metadata,
            "context": context or {},
        }
        self._check_history.append(entry)
        
        # Keep history limited
        if len(self._check_history) > 1000:
            self._check_history = self._check_history[-1000:]
