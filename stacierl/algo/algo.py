from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from ..replaybuffer import Batch
from .model import Model, ModelStateDicts
import torch


class Algo(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._device: torch.device = torch.device()

    @property
    @abstractmethod
    def model(self) -> Model:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abstractmethod
    def model_states(self) -> Dict:
        ...
        
    @property
    @abstractmethod
    def optimizer_state_dicts(self) -> Dict:
        ...

    @abstractmethod
    def update(self, batch: Batch) -> List[float]:
        ...

    @abstractmethod
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def set_model_states(self, model_states: Dict) -> None:
        ...
        
    @abstractmethod
    def set_optimizer_state_dicts(self, optimizer_state_dicts: Dict) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
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
