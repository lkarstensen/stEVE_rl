import torch.nn as nn
from abc import ABC, abstractmethod


class Network(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._initial_hidden_state = None

    @abstractmethod
    def copy(self):
        ...

    @property
    def initial_hidden_state(self):
        return self._initial_hidden_state
