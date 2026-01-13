from .broker_base import Broker

class PaperBroker(Broker):
    def __init__(self):
        self._store = {}
        self._id = 0

    def place(self, symbol:str, side:str, qty:int, order_type:str, price:float|None=None):
        self._id += 1
        oid = f"PB{self._id}"
        self._store[oid] = {"status":"filled","avg_price":price or 100.0,"filled_qty":qty}
        return {"order_id": oid, **self._store[oid]}

    def cancel(self, order_id:str) -> bool:
        return True

    def get_status(self, order_id:str):
        return self._store.get(order_id, {"status":"unknown"})
