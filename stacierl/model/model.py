from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import torch


class Model(ABC):
    @abstractmethod
    def get_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        ...

    @abstractmethod
    def to(self, device: torch.device):
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def copy_shared_memory(self):
        ...

    @property
    @abstractmethod
    def initial_hidden_state(self) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def all_state_dicts(self):
        ...

    @abstractmethod
    def all_parameters(self):
        ...

    @abstractmethod
    def load_all_state_dicts(self, all_state_dicts: dict):
        ...

    @abstractmethod
    def soft_tau_update_all(self, all_parameters: dict, tau: float):
        ...
