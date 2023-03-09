from copy import deepcopy
from typing import List, Tuple
import logging
from torch import nn
import torch

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

        self.layers: List[nn.Module] = []
        for in_size, out_size in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.ReLU())

        self.sequential = nn.Sequential(*self.layers)

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

    @property
    def device(self) -> torch.device:
        return self.layers[0].weight.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.sequential.forward(obs_batch)

        mean = self.mean(output)
        log_std = self.log_std(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...
