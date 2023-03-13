from copy import deepcopy
from typing import Any, Mapping
import torch
from .network import Network


class Dummy(Network):
    @property
    def n_inputs(self) -> int:
        return 0

    @property
    def n_outputs(self) -> int:
        return 0

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:

        return obs_batch

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        ...
