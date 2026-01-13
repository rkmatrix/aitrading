"""
Retry Handler with Exponential Backoff and Circuit Breaker
----------------------------------------------------------
Provides robust retry logic for broker API calls with circuit breaker pattern.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, TypeVar, Tuple
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes (half-open state)
    timeout_seconds: float = 60.0  # Time before attempting half-open
    reset_timeout: float = 300.0  # Time before resetting failure count


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by opening circuit when service is failing,
    and gradually testing recovery.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        self.logger = logging.getLogger("CircuitBreaker")
    
    def call(self, func: Callable[[], T], *args, **kwargs) -> Tuple[bool, Optional[T], Optional[str]]:
        """
        Execute function through circuit breaker.
        
        Returns:
            Tuple of (success: bool, result: T | None, error: str | None)
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                return False, None, "Circuit breaker is OPEN"
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return True, result, None
        except Exception as e:
            self._on_failure()
            return False, None, str(e)
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED (recovered)")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.warning("Circuit breaker OPEN (failed during half-open)")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()
                self.logger.error(
                    "Circuit breaker OPEN (failure threshold reached: %d)",
                    self.failure_count
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger.info("Circuit breaker manually reset")


class RetryHandler:
    """
    Retry handler with exponential backoff.
    
    Usage:
        handler = RetryHandler(config=RetryConfig())
        result = handler.execute(lambda: api_call())
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("RetryHandler")
    
    def execute(
        self,
        func: Callable[[], T],
        *args,
        **kwargs,
    ) -> Tuple[bool, Optional[T], Optional[str]]:
        """
        Execute function with retry logic.
        
        Returns:
            Tuple of (success: bool, result: T | None, error: str | None)
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(
                        "Function succeeded on attempt %d/%d",
                        attempt + 1,
                        self.config.max_retries + 1
                    )
                return True, result, None
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Don't retry on certain errors (e.g., validation errors)
                if self._should_not_retry(e):
                    self.logger.warning("Not retrying on error: %s", error_msg)
                    return False, None, error_msg
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        error_msg,
                        delay
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        "All %d attempts failed. Last error: %s",
                        self.config.max_retries + 1,
                        error_msg
                    )
        
        return False, None, str(last_error) if last_error else "Unknown error"
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay
    
    def _should_not_retry(self, error: Exception) -> bool:
        """Check if error should not be retried."""
        error_str = str(error).lower()
        
        # Don't retry on validation/authentication errors
        non_retryable = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid",
            "validation",
            "missing",
        ]
        
        return any(keyword in error_str for keyword in non_retryable)


class RetryableBrokerCall:
    """
    Combines retry handler and circuit breaker for broker calls.
    
    Usage:
        retryable = RetryableBrokerCall(
            retry_config=RetryConfig(max_retries=3),
            circuit_config=CircuitBreakerConfig(failure_threshold=5)
        )
        success, result, error = retryable.execute(lambda: broker.submit_order(...))
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.retry_handler = RetryHandler(retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.logger = logging.getLogger("RetryableBrokerCall")
    
    def execute(self, func: Callable[[], T], *args, **kwargs) -> Tuple[bool, Optional[T], Optional[str]]:
        """
        Execute function through circuit breaker and retry logic.
        
        Returns:
            Tuple of (success: bool, result: T | None, error: str | None)
        """
        # First check circuit breaker
        success, result, error = self.circuit_breaker.call(func, *args, **kwargs)
        
        if not success:
            if error == "Circuit breaker is OPEN":
                return False, None, error
            # If circuit breaker allowed call but it failed, retry
            return self.retry_handler.execute(func, *args, **kwargs)
        
        return True, result, None
