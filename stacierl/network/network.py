from abc import abstractmethod
import torch.nn as nn


import torch


class Network(nn.Module):
    @property
    @abstractmethod
    def n_inputs(self) -> int:
        ...

    @n_inputs.setter
    @abstractmethod
    def n_inputs(self, n_inputs: int) -> None:
        ...

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        ...

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
    def copy(self):
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
