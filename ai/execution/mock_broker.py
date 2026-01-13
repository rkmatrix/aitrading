"""
MockBroker — Simulated broker for Phase 71.
Does not talk to Alpaca or any real API.
"""

class MockBroker:
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: float = None,
        tag: str = None,
        client_order_id: str = None,
        is_flattening: bool = False,
    ):
        """
        Returns a fake successful order response.
        """
        return {
            "id": f"mock-{client_order_id}",
            "filled_avg_price": limit_price or 190.25,
            "status": "filled",
        }
