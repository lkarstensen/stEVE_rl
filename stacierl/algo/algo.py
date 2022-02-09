from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from ..replaybuffer import Batch
from .model import Model, PytorchStatesContainer
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
    def network_states_container(self) -> PytorchStatesContainer:
        ...

    @property
    @abstractmethod
    def optimizer_states_container(self) -> PytorchStatesContainer:
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
    def set_network_states(self, network_states_container: PytorchStatesContainer) -> None:
        ...

    @abstractmethod
    def set_optimizer_states(self, optimizer_states_container: PytorchStatesContainer) -> None:
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
