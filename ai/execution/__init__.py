"""
ai.execution package initializer.
Provides unified imports for execution-related modules.
"""
from __future__ import annotations

"""
ai.execution package
"""

from ai.execution.smart_order_router import SmartOrderRouter
from ai.execution.broker_alpaca_live import AlpacaLiveBroker
from ai.execution.multi_exchange_router import MultiExchangeRouter


__all__ = ["SmartOrderRouter", "AlpacaLiveExecutor"]
