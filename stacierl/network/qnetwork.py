import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import PackedSequence
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

        self.layers = nn.ModuleList()
        for input, output in zip(layers_in, layers_out):
            self.layers.append(nn.Linear(input, output))

        self.layers.append(nn.Linear(hidden_layers[-1], 1))

        # init weights and bias
        # for i in range(len(self.layers)):
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

    def forward(
        self,
        state_batch: PackedSequence,
        action_batch: PackedSequence,
    ) -> torch.Tensor:
        state = state_batch.data
        action = action_batch.data
        input = torch.cat([state, action], dim=1)
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
