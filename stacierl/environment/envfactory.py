from abc import ABC, abstractmethod
from .environment import Environment


class EnvFactory(ABC):
    @abstractmethod
    def create_env(self) -> Environment:
        pass
