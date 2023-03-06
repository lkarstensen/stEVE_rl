from copy import deepcopy
import torch
from .network import Network


class Dummy(Network):
    def __init__(self, n_inputs: int):
        super().__init__()
        self._n_inputs = n_inputs

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_inputs

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return input_batch

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...
