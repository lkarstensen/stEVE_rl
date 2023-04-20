from abc import abstractmethod
from typing import List, Union
from torch import nn
import torch
from ...util import EveRLObject


class Component(nn.Module, EveRLObject):
    n_inputs: int
    n_outputs: int
    output_layer_size: int
    device: torch.device

    @abstractmethod
    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        ...

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        with torch.no_grad():
            output = self.forward(*args, obs_batch=obs_batch, **kwds)
        return output

    @abstractmethod
    def reset(self) -> None:
        ...
