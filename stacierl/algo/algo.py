from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List, Dict, Optional
import numpy as np
import torch
from ..replaybuffer import Batch
from .model import Model
from ..util import ConfigHandler


class Algo(ABC):
    model: Model
    device: torch.device

    lr_scheduler_step_counter = 0

    def state_dicts_network(
        self, destination: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.model.state_dicts_network(destination)

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        return self.model.load_state_dicts_network(state_dicts)

    @abstractmethod
    def update(self, batch: Batch) -> List[float]:
        ...

    @abstractmethod
    def lr_scheduler_step(self) -> None:
        self.lr_scheduler_step_counter += 1

    @abstractmethod
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def copy(self):
        copy = deepcopy(self)
        return copy

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def close(self):
        ...

    def save_config(self, file_path: str):
        confighandler = ConfigHandler()
        confighandler.save_config(self, file_path)
