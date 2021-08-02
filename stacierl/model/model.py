from abc import ABC, abstractmethod
import torch


class Model(ABC):
    @abstractmethod
    def to(self, device: torch.device):
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def copy_shared_memory(self):
        ...
