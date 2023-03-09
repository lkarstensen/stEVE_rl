from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
import torch
from ..util import ConfigHandler


class Model(ABC):
    device: torch.device

    @abstractmethod
    def state_dicts_network(
        self, destination: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_play_action(self, flat_state: np.ndarray, evaluation: bool) -> np.ndarray:
        ...

    def to(self, device: torch.device):
        self.device = device

    def copy(self):
        copy = deepcopy(self)
        return copy

    @abstractmethod
    def copy_play_only(self):
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def close(self):
        ...

    def save_config(self, file_path: str):
        confighandler = ConfigHandler()
        confighandler.save_config(self, file_path)
