"""
broker_dry.py — simple DRY-RUN broker adapter with an async execute()
Used as a safe fallback whenever a real broker isn't available.
"""

import asyncio, time, random

async def execute(order: dict) -> dict:
    # Simulate network + exchange latency
    await asyncio.sleep(random.uniform(0.05, 0.20))
    return {
        "symbol": order.get("symbol"),
        "order_id": f"DRY_{int(time.time()*1000)}",
        "filled_qty": float(order.get("qty", 0)),
        "side": order.get("side"),
        "pnl": random.uniform(-0.05, 0.05),
        "slippage": random.uniform(0, 0.001),
        "status": "filled",
    }

broker = {"execute": execute, "name": "dry_stub", "mode": "DRY_RUN"}
