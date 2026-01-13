"""
Validation Script for AITradingBot Improvements
------------------------------------------------
Validates that all improvements are properly implemented and working.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Validation")


def test_imports():
    """Test that all new modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from tools.env_loader import validate_api_keys
        logger.info("✅ tools.env_loader imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import tools.env_loader: {e}")
        return False
    
    try:
        from ai.execution.order_validator import OrderValidator, ValidationConfig
        logger.info("✅ ai.execution.order_validator imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.execution.order_validator: {e}")
        return False
    
    try:
        from ai.execution.retry_handler import RetryHandler, CircuitBreaker, RetryableBrokerCall
        logger.info("✅ ai.execution.retry_handler imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.execution.retry_handler: {e}")
        return False
    
    try:
        from ai.guardian.kill_switch_monitor import KillSwitchMonitor, KillSwitchConfig
        logger.info("✅ ai.guardian.kill_switch_monitor imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.guardian.kill_switch_monitor: {e}")
        return False
    
    try:
        from ai.market.market_clock import MarketClock
        logger.info("✅ ai.market.market_clock imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.market.market_clock: {e}")
        return False
    
    try:
        from ai.market.market_data_validator import MarketDataValidator, MarketDataConfig
        logger.info("✅ ai.market.market_data_validator imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.market.market_data_validator: {e}")
        return False
    
    try:
        from ai.execution.broker_adapter import BrokerAdapter, create_broker_adapter
        logger.info("✅ ai.execution.broker_adapter imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.execution.broker_adapter: {e}")
        return False
    
    try:
        from ai.risk.risk_manager import RiskManager, RiskCheckResult
        logger.info("✅ ai.risk.risk_manager imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.risk.risk_manager: {e}")
        return False
    
    try:
        from ai.monitor.reconciliation_scheduler import ReconciliationScheduler, ReconciliationSchedule
        logger.info("✅ ai.monitor.reconciliation_scheduler imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.monitor.reconciliation_scheduler: {e}")
        return False
    
    try:
        from ai.utils.config_validator import ConfigValidator, validate_startup_config
        logger.info("✅ ai.utils.config_validator imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.utils.config_validator: {e}")
        return False
    
    try:
        from ai.logging.structured_logger import StructuredLogger
        logger.info("✅ ai.logging.structured_logger imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ai.logging.structured_logger: {e}")
        return False
    
    return True


def test_order_validator():
    """Test OrderValidator functionality."""
    logger.info("Testing OrderValidator...")
    
    try:
        from ai.execution.order_validator import OrderValidator, ValidationConfig
        
        config = ValidationConfig(
            min_qty=1.0,
            max_qty=1000.0,
            min_notional=100.0,
        )
        validator = OrderValidator(config=config)
        
        # Test valid order
        order = {
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 10,
            "order_type": "market",
        }
        is_valid, error = validator.validate(order)
        if not is_valid:
            logger.error(f"❌ Valid order rejected: {error}")
            return False
        
        # Test invalid symbol
        order["symbol"] = "INVALID_SYMBOL_TOO_LONG"
        is_valid, error = validator.validate(order)
        if is_valid:
            logger.error("❌ Invalid symbol accepted")
            return False
        
        logger.info("✅ OrderValidator tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ OrderValidator test failed: {e}", exc_info=True)
        return False


def test_market_clock():
    """Test MarketClock functionality."""
    logger.info("Testing MarketClock...")
    
    try:
        from ai.market.market_clock import MarketClock
        
        clock = MarketClock()
        
        # Test basic functionality
        is_open = clock.is_open()
        status = clock.get_market_status()
        
        logger.info(f"✅ MarketClock working - Market open: {is_open}")
        logger.info(f"   Status: {status}")
        
        return True
    except Exception as e:
        logger.error(f"❌ MarketClock test failed: {e}", exc_info=True)
        return False


def test_retry_handler():
    """Test RetryHandler functionality."""
    logger.info("Testing RetryHandler...")
    
    try:
        from ai.execution.retry_handler import RetryHandler, RetryConfig, CircuitBreaker, CircuitBreakerConfig
        
        # Test retry handler
        config = RetryConfig(max_retries=2, initial_delay=0.1)
        handler = RetryHandler(config=config)
        
        # Test successful call
        def success_func():
            return "success"
        
        success, result, error = handler.execute(success_func)
        if not success or result != "success":
            logger.error("❌ RetryHandler failed on success case")
            return False
        
        # Test circuit breaker
        circuit_config = CircuitBreakerConfig(failure_threshold=2)
        circuit = CircuitBreaker(circuit_config)
        
        logger.info("✅ RetryHandler tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ RetryHandler test failed: {e}", exc_info=True)
        return False


def test_kill_switch_monitor():
    """Test KillSwitchMonitor functionality."""
    logger.info("Testing KillSwitchMonitor...")
    
    try:
        from ai.guardian.kill_switch_monitor import KillSwitchMonitor, KillSwitchConfig
        
        config = KillSwitchConfig(
            flag_path="data/runtime/test_kill_switch.flag",
            heartbeat_interval_sec=30.0,
        )
        monitor = KillSwitchMonitor(config)
        
        # Test check
        state_info = monitor.check()
        logger.info(f"✅ KillSwitchMonitor working - Active: {state_info['active']}")
        
        # Cleanup test file if created
        test_flag = Path("data/runtime/test_kill_switch.flag")
        if test_flag.exists():
            test_flag.unlink()
        
        return True
    except Exception as e:
        logger.error(f"❌ KillSwitchMonitor test failed: {e}", exc_info=True)
        return False


def test_broker_adapter():
    """Test BrokerAdapter functionality."""
    logger.info("Testing BrokerAdapter...")
    
    try:
        from ai.execution.broker_adapter import create_broker_adapter, DummyBrokerAdapter
        
        # Test dummy broker (doesn't require real API keys)
        broker = DummyBrokerAdapter(mode="DEMO")
        
        account = broker.get_account()
        if account.equity <= 0:
            logger.error("❌ DummyBrokerAdapter returned invalid account")
            return False
        
        positions = broker.get_positions()
        logger.info(f"✅ BrokerAdapter working - Account equity: ${account.equity:,.2f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ BrokerAdapter test failed: {e}", exc_info=True)
        return False


def test_config_validator():
    """Test ConfigValidator functionality."""
    logger.info("Testing ConfigValidator...")
    
    try:
        from ai.utils.config_validator import ConfigValidator
        
        validator = ConfigValidator()
        validator.add_rule("TEST_KEY", required=True)
        
        # Test with missing key
        is_valid, errors = validator.validate({"OTHER_KEY": "value"})
        if is_valid:
            logger.error("❌ ConfigValidator accepted missing required key")
            return False
        
        # Test with present key
        is_valid, errors = validator.validate({"TEST_KEY": "value"})
        if not is_valid:
            logger.error(f"❌ ConfigValidator rejected valid config: {errors}")
            return False
        
        logger.info("✅ ConfigValidator tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ ConfigValidator test failed: {e}", exc_info=True)
        return False


def test_structured_logger():
    """Test StructuredLogger functionality."""
    logger.info("Testing StructuredLogger...")
    
    try:
        from ai.logging.structured_logger import StructuredLogger
        
        logger_instance = StructuredLogger(
            name="TestLogger",
            console=False,  # Don't spam console during tests
        )
        
        logger_instance.log_order(
            "Test order",
            symbol="AAPL",
            side="BUY",
            qty=10,
            price=150.0,
        )
        
        logger.info("✅ StructuredLogger tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ StructuredLogger test failed: {e}", exc_info=True)
        return False


def test_file_structure():
    """Test that all expected files exist."""
    logger.info("Testing file structure...")
    
    expected_files = [
        ".gitignore",
        "ai/execution/order_validator.py",
        "ai/execution/retry_handler.py",
        "ai/execution/broker_adapter.py",
        "ai/guardian/kill_switch_monitor.py",
        "ai/market/market_clock.py",
        "ai/market/market_data_validator.py",
        "ai/risk/risk_manager.py",
        "ai/monitor/reconciliation_scheduler.py",
        "ai/utils/config_validator.py",
        "ai/logging/structured_logger.py",
        "tests/test_order_validator.py",
        "tests/test_config_validator.py",
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    logger.info("✅ All expected files exist")
    return True


def main():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("AITradingBot Improvements Validation")
    logger.info("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("OrderValidator", test_order_validator),
        ("MarketClock", test_market_clock),
        ("RetryHandler", test_retry_handler),
        ("KillSwitchMonitor", test_kill_switch_monitor),
        ("BrokerAdapter", test_broker_adapter),
        ("ConfigValidator", test_config_validator),
        ("StructuredLogger", test_structured_logger),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        logger.info(f"Running: {test_name}")
        logger.info("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All improvements validated successfully!")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
