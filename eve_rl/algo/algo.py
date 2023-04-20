from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import numpy as np
import torch
from ..replaybuffer import Batch
from ..model import Model, ModelPlayOnly
from ..util import EveRLObject


class AlgoPlayOnly(EveRLObject, ABC):
    model: ModelPlayOnly
    device: torch.device

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        return self.model.load_state_dicts_network(state_dicts)

    @abstractmethod
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def close(self):
        ...


class Algo(EveRLObject, ABC):
    model: Model
    device: torch.device

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
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def to_play_only(self) -> AlgoPlayOnly:
        ...

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def close(self):
        ...
