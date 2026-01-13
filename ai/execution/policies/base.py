from abc import ABC, abstractmethod
from typing import Dict, Any

class ExecutionPolicy(ABC):
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return child order {size:int, price:float|None, aggression:str}"""
        ...
