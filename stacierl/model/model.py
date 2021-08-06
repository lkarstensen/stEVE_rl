from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ModelStateDicts(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def copy(self):
        ...

    def to(self, device: torch.device):
        for state_dict in self:
            for tensor_name, tensor in state_dict.items():
                state_dict[tensor_name] = tensor.to(device)

    def __add__(self, other):
        copy = self.copy()
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in copy:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor + other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(copy, other):
                for own_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    own_tensor.data.copy_(own_tensor + other_tensor)
        return copy

    def __iadd__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in self:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor + other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(self, other):
                for own_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    own_tensor.data.copy_(own_tensor + other_tensor)
        return self

    def __mul__(self, other):
        copy = self.copy()
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in copy:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor * other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(copy, other):
                for copy_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    copy_tensor.data.copy_(copy_tensor * other_tensor)
        return copy

    def __imul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in self:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor * other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(self, other):
                for copy_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    copy_tensor.data.copy_(copy_tensor * other_tensor)
        return self

    def __truediv__(self, other):
        copy = self.copy()
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in copy:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor / other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(copy, other):
                for copy_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    copy_tensor.data.copy_(copy_tensor / other_tensor)
        return copy

    def __itruediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            for state_dict in self:
                for tensor in state_dict.values():
                    tensor.data.copy_(tensor / other)
        if isinstance(other, self.__class__):
            for copy_state_dict, other_state_dict in zip(self, other):
                for copy_tensor, other_tensor in zip(
                    copy_state_dict.values(), other_state_dict.values()
                ):
                    copy_tensor.data.copy_(copy_tensor / other_tensor)
        return self


@dataclass
class ModelNetworks(ABC):
    @abstractmethod
    def to(self, device: torch.device) -> None:
        ...

    @abstractmethod
    def soft_tau_update(self, model_parameters, tau: float):
        ...

    @abstractmethod
    def load_state_dicts(self, model_state_dicts):
        ...

    @property
    @abstractmethod
    def state_dicts(self):
        ...

    @abstractmethod
    def __iter__(self):
        ...

    def to(self, device: torch.device) -> None:
        self.device = device
        for net in self:
            net.to(device)


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

    @property
    @abstractmethod
    def nets(self) -> ModelNetworks:
        ...
