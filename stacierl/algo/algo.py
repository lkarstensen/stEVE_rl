from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple
import numpy as np
from ..replaybuffer import Batch


class Algo(ABC):
    @abstractmethod
    def update(self, batch: Batch) -> None:
        ...

    @abstractmethod
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save_model(self, path):
        ...

    @abstractmethod
    def load_model(self, path):
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def copy_shared_memory(self):
        ...
