import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from .network import Network


class MLP(Network):
    def __init__(self, hidden_layers: List[int]):
        super().__init__()

        self.hidden_layers = hidden_layers
        layers_in = hidden_layers[:-1]
        layers_out = hidden_layers[1:]

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_in, layers_out):
            self.layers.append(nn.Linear(input, output))

    @property
    def n_inputs(self) -> int:
        return self.layers[0].in_features

    @property
    def n_outputs(self) -> int:
        return self.layers[-1].out_features

    @property
    def input_is_set(self) -> bool:
        return len(self.layers) == len(self.hidden_layers)

    def set_input(self, n_input):
        n_output = self.hidden_layers[0]
        self.layers.insert(0, nn.Linear(n_input, n_output))

        # weight init
        for layer in self.layers[:-1]:
            # torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("relu"))
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.0)

        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain("linear"))
        nn.init.constant_(self.layers[-1].bias, 0.0)

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

        copy = self.__class__(self.hidden_layers)
        return copy

    def reset(self) -> None:
        ...
