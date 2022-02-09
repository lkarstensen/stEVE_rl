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
    def model_states_container(self) -> ModelStateDicts:
        ...
        
    @property
    @abstractmethod
    def optimizer_states_container(self) -> ModelStateDicts:
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
    def set_model_states(self, model_states_container: ModelStateDicts, continue_training: bool = True) -> None:
        ...
        
    @abstractmethod
    def set_optimizer_states(self, optimizer_states_container: ModelStateDicts) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
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
