from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Network


class MLP(Network):
    def __init__(self, n_inputs, hidden_layers: List[int]):
        super().__init__()
        self._n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        layers_in = [n_inputs] + hidden_layers[:-1]
        layers_out = hidden_layers

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_in, layers_out):
            self.layers.append(nn.Linear(input, output))

        # weight init
        for layer in self.layers[:-1]:
            # torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("relu"))
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.0)

        nn.init.xavier_uniform_(
            self.layers[-1].weight, gain=nn.init.calculate_gain("linear")
        )
        nn.init.constant_(self.layers[-1].bias, 0.0)

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self.layers[-1].out_features

    @property
    def device(self) -> torch.device:
        return self.layers[0].weight.device

    def forward(self, input_batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        input = input_batch
        for layer in self.layers[:-1]:
            output = layer(input)
            output = F.relu(output)
            input = output

        # output without relu
        output = self.layers[-1](input)

        return output

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...
