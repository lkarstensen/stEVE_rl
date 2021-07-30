from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np


class ActionSpace(ABC):
    @property
    @abstractmethod
    def shape(self) -> Tuple[float]:
        ...

    @property
    @abstractmethod
    def low(self) -> Tuple[float]:
        ...

    @property
    @abstractmethod
    def high(self) -> Tuple[float]:
        ...


class ObservationSpace(ABC):
    @property
    @abstractmethod
    def shape(self) -> Dict[str, Tuple[float]]:
        ...

    @property
    @abstractmethod
    def low(self) -> Dict[str, np.ndarray]:
        ...

    @property
    @abstractmethod
    def high(self) -> Dict[str, np.ndarray]:
        ...


class Environment(ABC):
    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict[str, Any]]]:
        ...

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        ...

    @abstractmethod
    def render(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> ObservationSpace:
        ...
