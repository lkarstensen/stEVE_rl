from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple, Dict
import numpy as np
import torch
from dataclasses import dataclass

from ..network import Network


@dataclass
class ModelStateDicts(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def copy(self):
        ...
        
    @abstractmethod
    def to_dict(self):
        ...
        
    @abstractmethod
    def from_dict(self, model_state_dict: Dict):
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


class Model(ABC):
    @property
    @abstractmethod
    def model_states_container(self) -> ModelStateDicts:
        ...
        
    @property
    @abstractmethod
    def optimizer_state_dicts(self) -> Dict:
        ...

    @abstractmethod
    def get_play_action(self, flat_state: np.ndarray) -> np.ndarray:
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

    @abstractmethod
    def set_model_states(self, model_states_container: ModelStateDicts, continue_training: bool = True) -> None:
        ...
        
    @abstractmethod
    def set_optimizer_state_dicts(self, optimizer_state_dicts: Dict) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
