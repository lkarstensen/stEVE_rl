from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .component import Component


class MLP(Component):
    def __init__(
        self,
        hidden_layers: List[int],
        n_inputs: Optional[int] = None,
        output_layer_size: Optional[Union[int, List[int]]] = None,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.init_w = init_w

        self._input_layer: nn.Linear = None
        self._output_layers: List[nn.Linear] = None
        self._layers: List[nn.Linear] = nn.ModuleList()
        layers_in = self.hidden_layers[:-1]
        layers_out = self.hidden_layers[1:]
        for in_size, out_size in zip(layers_in, layers_out):
            self._layers.append(nn.Linear(in_size, out_size))
        if n_inputs is not None:
            self.n_inputs = n_inputs
        if output_layer_size is not None:
            self.output_layer_size = output_layer_size

    @property
    def n_inputs(self) -> int:
        if self._input_layer is None:
            return None
        return self._input_layer.in_features

    @n_inputs.setter
    def n_inputs(self, n_inputs: int) -> None:
        if self._input_layer is None:
            self._input_layer = nn.Linear(n_inputs, self._layers[0].in_features)

        elif self.n_inputs != n_inputs:
            raise ValueError(f"{self.n_inputs=} already set different than {n_inputs=}")

    @property
    def n_outputs(self) -> Union[int, List[int]]:
        if self._output_layers is None:
            return self._layers[-1].out_features
        out = [layer.out_features for layer in self._output_layers]
        out = out[0] if len(out) == 1 else out
        return out

    @property
    def output_layer_size(self) -> Union[int, List[int]]:
        if self._output_layers is None:
            return None
        return self.n_outputs

    @output_layer_size.setter
    def output_layer_size(self, output_layer_size: Union[int, List[int]]) -> None:
        if self._output_layers is None:
            if isinstance(output_layer_size, int):
                output_layer_size = [output_layer_size]
            self._output_layers = nn.ModuleList()
            for n_outputs in output_layer_size:
                layer = nn.Linear(self._layers[-1].out_features, n_outputs)
                layer.weight.data.uniform_(-self.init_w, self.init_w)
                layer.bias.data.uniform_(-self.init_w, self.init_w)
                self._output_layers.append(layer)
        elif self.output_layer_size != output_layer_size:
            raise ValueError(
                f"{self.output_layer_size=} already set different than {output_layer_size=}"
            )

    @property
    def device(self) -> torch.device:  # pylint: disable=no-member
        return self._layers[0].weight.device

    # TODO: Add F.relu after input layer
    def forward(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        state = self._input_layer(obs_batch)
        for layer in self._layers:
            state = layer(state)
            state = F.relu(state)
        if self._output_layers is not None:
            # output without relu
            state = [layer(state) for layer in self._output_layers]
            state = state[0] if len(state) == 1 else state
        return state

    def reset(self) -> None:
        ...
