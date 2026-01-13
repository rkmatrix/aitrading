from abc import ABC, abstractmethod
from typing import Dict, Any

class Broker(ABC):
    @abstractmethod
    def place(self, symbol:str, side:str, qty:int, order_type:str, price:float|None=None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def cancel(self, order_id:str) -> bool:
        ...

    @abstractmethod
    def get_status(self, order_id:str) -> Dict[str, Any]:
        ...
