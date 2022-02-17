from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple, Dict
import numpy as np
import torch
from dataclasses import dataclass

from stacierl.util.stacierluserobject import StacieRLUserObject

from ..network import Network


@dataclass
class NetworkStatesContainer(ABC):
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
    def from_dict(self, state_dict: Dict):
        ...

    def to(self, device: torch.device):
        for network in self:
            for tensor_name, tensor in network.items():
                network[tensor_name] = tensor.to(device)

    def __add__(self, other):
        copy = self.copy()
        for copy_network, other_network in zip(copy, other):
            for copy_tensor, other_tensor in zip(copy_network.values(), other_network.values()):
                copy_tensor.data.copy_(copy_tensor + other_tensor)
        return copy

    def __iadd__(self, other):
        for self_network, other_network in zip(self, other):
            for own_tensor, other_tensor in zip(self_network.values(), other_network.values()):
                own_tensor.data.copy_(own_tensor + other_tensor)
        return self

    def __mul__(self, other):
        copy = self.copy()
        for network in copy:
            for tensor in network.values():
                tensor.data.copy_(tensor * other)
        return copy

    def __imul__(self, other):
        for network in self:
            for tensor in network.values():
                tensor.data.copy_(tensor * other)
        return self

    def __truediv__(self, other):
        copy = self.copy()

        for network in copy:
            for tensor in network.values():
                tensor.data.copy_(tensor / other)
        return copy

    def __itruediv__(self, other):
        for network in self:
            for tensor in network.values():
                tensor.data.copy_(tensor / other)

        return self


@dataclass
class OptimizerStatesContainer(ABC):
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
    def from_dict(self, state_dict: Dict):
        ...

    def to(self, device: torch.device):
        for optimizer in self:
            for param_nr, param in optimizer["state"].items():
                optimizer["state"][param_nr]["exp_avg"] = param["exp_avg"].to(device)
                optimizer["state"][param_nr]["exp_avg_sq"] = param["exp_avg_sq"].to(device)

    def __add__(self, other):
        copy = self.copy()
        for copy_optimizer, other_optimizer in zip(copy, other):
            for param_nr in copy_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    copy_optimizer["state"][param_nr][key] = (
                        copy_optimizer["state"][param_nr][key]
                        + other_optimizer["state"][param_nr][key]
                    )
        return copy

    def __iadd__(self, other):
        for self_optimizer, other_optimizer in zip(self, other):
            for param_nr in self_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    self_optimizer["state"][param_nr][key] = (
                        self_optimizer["state"][param_nr][key]
                        + other_optimizer["state"][param_nr][key]
                    )
        return self

    def __mul__(self, other):
        copy = self.copy()
        for copy_optimizer in copy:
            for param_nr in copy_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    copy_optimizer["state"][param_nr][key] = (
                        copy_optimizer["state"][param_nr][key] * other
                    )
        return copy

    def __imul__(self, other):
        for self_optimizer in self:
            for param_nr in self_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    self_optimizer["state"][param_nr][key] = (
                        self_optimizer["state"][param_nr][key] * other
                    )
        return self

    def __truediv__(self, other):
        copy = self.copy()
        for copy_optimizer in copy:
            for param_nr in copy_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    copy_optimizer["state"][param_nr][key] = (
                        copy_optimizer["state"][param_nr][key] / other
                    )
        return copy

    def __itruediv__(self, other):
        for self_optimizer in self:
            for param_nr in self_optimizer["state"].keys():
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    self_optimizer["state"][param_nr][key] = (
                        self_optimizer["state"][param_nr][key] / other
                    )
        return self


class Model(StacieRLUserObject, ABC):
    @property
    @abstractmethod
    def network_states_container(self) -> NetworkStatesContainer:
        ...

    @property
    @abstractmethod
    def optimizer_states_container(self) -> OptimizerStatesContainer:
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
    def set_network_states(self, network_states_container: NetworkStatesContainer) -> None:
        ...

    @abstractmethod
    def set_optimizer_states(self, optimizer_states_container: OptimizerStatesContainer) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
