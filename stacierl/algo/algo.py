from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple
import numpy as np
from ..replaybuffer import Batch
import torch


class Algo(ABC):
    @abstractmethod
    def update(self, batch: Batch) -> None:
        ...

    @abstractmethod
    def get_exploration_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        ...

    @abstractmethod
    def get_eval_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        ...

    @abstractmethod
    def get_initial_hidden_state(self):
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
