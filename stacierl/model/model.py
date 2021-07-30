from abc import ABC, abstractmethod
import torch


class Model(ABC):
    @abstractmethod
    def to(self, device: torch.device):
        ...
