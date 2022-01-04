import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from .network import Network


class QNetwork(Network):
    def __init__(self, hidden_layers: List[int], init_w=3e-3):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.init_w = init_w
        self._n_observations = None
        self._n_actions = None

        layers_in = hidden_layers[:-1]
        layers_out = hidden_layers[1:]

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_in, layers_out):
            self.layers.append(nn.Linear(input, output))

        self.layers.append(nn.Linear(hidden_layers[-1], 1))

        self.layers[-1].weight.data.uniform_(-self.init_w, self.init_w)
        self.layers[-1].bias.data.uniform_(-self.init_w, self.init_w)

    @property
    def input_is_set(self) -> bool:
        return self._n_observations is not None

    @property
    def n_inputs(self) -> Tuple[int, int]:
        return self._n_observations, self._n_actions

    @property
    def n_outputs(self) -> int:
        return 1

    def set_input(self, n_observations, n_actions):
        self._n_observations = n_observations
        self._n_actions = n_actions
        n_input = n_observations + n_actions
        n_output = self.hidden_layers[0]
        self.layers.insert(0, nn.Linear(n_input, n_output))

        """
        # init weights and bias
        for layer in self.layers[:-1]:
            # torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("relu"))
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.0)

        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain("linear"))
        nn.init.constant_(self.layers[-1].bias, 0.0)
        """

    def forward(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:

        input = torch.dstack([state_batch, action_batch])
        for layer in self.layers[:-1]:
            output = layer(input)
            output = F.relu(output)
            input = output

        # output without relu
        q_value_batch = self.layers[-1](output)

        return q_value_batch

    def copy(self):

        copy = self.__class__(self.hidden_layers, self.init_w)
        return copy

    def reset(self) -> None:
        ...
