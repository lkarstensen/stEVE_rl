import torch.nn as nn
from abc import ABC, abstractmethod

from torch.nn.utils.rnn import PackedSequence


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
    def forward(self, input: PackedSequence) -> PackedSequence:
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
