# ai/execution/execution_strategy.py
import numpy as np
import datetime as dt

class ExecutionStrategy:
    """Strategy for slicing and scheduling orders."""

    def __init__(self, strategy_type="TWAP", total_qty=100, duration_minutes=30):
        self.strategy_type = strategy_type
        self.total_qty = total_qty
        self.duration = duration_minutes
        self.start_time = dt.datetime.now()

    def get_next_slice(self, elapsed_minutes):
        if self.strategy_type == "TWAP":
            return self.total_qty / (self.duration or 1)
        elif self.strategy_type == "VWAP":
            weight = np.sin(np.pi * elapsed_minutes / self.duration)
            return self.total_qty * weight / np.sum(weight)
        else:
            return self.total_qty
