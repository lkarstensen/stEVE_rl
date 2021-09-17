from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from ..replaybuffer import Batch
from .. import model
import torch


class Algo(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._device: torch.device = torch.device()

    @property
    @abstractmethod
    def model(self) -> model.Model:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def update(self, batch: Batch) -> List[float]:
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

    @abstractmethod
    def to(self, device: torch.device):
        ...
