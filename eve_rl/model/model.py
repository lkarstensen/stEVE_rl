from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from ..util import EveRLObject


class ModelPlayOnly(EveRLObject, ABC):
    device: torch.device

    @abstractmethod
    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        ...

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def close(self):
        ...


class Model(EveRLObject, ABC):
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
    def state_dicts_optimizer(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dicts_optimizer(self, state_dicts: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def state_dicts_scheduler(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dicts_scheduler(self, state_dicts: Dict[str, Any]) -> None:
        ...

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def to_play_only(self) -> ModelPlayOnly:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def close(self):
        ...
