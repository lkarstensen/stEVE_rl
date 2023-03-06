from copy import deepcopy
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Network


class QNetwork(Network):
    def __init__(
        self, n_observations: int, n_actions: int, hidden_layers: List[int], init_w=3e-3
    ):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers
        self.init_w = init_w

        n_input = n_observations + n_actions
        layers_in = [n_input] + hidden_layers[:-1]
        layers_out = hidden_layers

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_in, layers_out):
            self.layers.append(nn.Linear(input, output))

        self.layers.append(nn.Linear(hidden_layers[-1], 1))

        self.layers[-1].weight.data.uniform_(-self.init_w, self.init_w)
        self.layers[-1].bias.data.uniform_(-self.init_w, self.init_w)

    @property
    def n_inputs(self) -> Tuple[int, int]:
        return self.n_observations, self.n_actions

    @property
    def n_outputs(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        return self.layers[0].weight.device

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

        return deepcopy(self)

    def reset(self) -> None:
        ...
