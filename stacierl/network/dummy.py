import torch
from .network import Network


class Dummy(Network):
    def __init__(self):
        super().__init__()
        self._n_inputs = None

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_inputs

    @property
    def input_is_set(self) -> bool:
        return self._n_inputs is not None

    def set_input(self, n_inputs):
        self._n_inputs = n_inputs

    def forward(self, input_batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return input_batch

    def copy(self):

        copy = self.__class__()
        return copy

    def reset(self) -> None:
        ...
