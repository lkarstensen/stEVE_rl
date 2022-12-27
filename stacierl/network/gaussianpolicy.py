from copy import deepcopy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .network import Network


class GaussianPolicy(Network):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_layers: List[int],
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean = None
        self.log_std = None

        layers_input = [n_observations] + hidden_layers[:-1]
        layers_output = hidden_layers

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(input, output))

        init_w = 3e-3
        last_output = self.hidden_layers[-1]
        self.mean = nn.Linear(last_output, n_actions)
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)

        self.log_std = nn.Linear(last_output, n_actions)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    @property
    def n_inputs(self) -> int:
        return self.n_observations

    @property
    def n_outputs(self) -> Tuple[int, int]:
        return self.n_actions, self.n_actions

    def forward(
        self, state_batch: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = state_batch
        for layer in self.layers:
            output = layer(input)
            output = F.relu(output)
            input = output

        mean = self.mean(output)
        log_std = self.log_std(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...
