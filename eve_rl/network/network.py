from abc import abstractmethod
from torch import nn
import torch

from ..util import EveRLObject


class Network(nn.Module, EveRLObject):
    n_observations: int
    n_actions: int

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def forward(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        ...

    def forward_play(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(*args, obs_batch=obs_batch, **kwds)
        return output

    @abstractmethod
    def reset(self) -> None:
        ...
