import torch.nn as nn
from abc import ABC, abstractmethod


import torch


class Network(ABC, nn.Module):
    @property
    @abstractmethod
    def input_is_set(self) -> bool:
        ...

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        ...

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        ...

    @abstractmethod
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def set_input(self, n_inputs: int):
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
