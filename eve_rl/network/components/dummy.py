from typing import Any, Mapping, Optional
import torch
from .component import Component


class Dummy(Component):
    def __init__(self, n_inputs: Optional[int] = None) -> None:
        super().__init__()
        self.n_inputs = n_inputs

    @property
    def n_outputs(self) -> int:
        return self.n_inputs

    @property
    def output_layer_size(self) -> int:
        return None

    @property
    def device(self) -> torch.device:  # pylint: disable=no-member
        return None

    # pylint: disable=arguments-differ
    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return obs_batch

    def reset(self) -> None:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        ...
