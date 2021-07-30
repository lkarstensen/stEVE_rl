import torch.nn as nn
from abc import ABC, abstractmethod


class Network(ABC, nn.Module):
    @abstractmethod
    def copy(self):
        ...
