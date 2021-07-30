import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
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
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.hidden_layers = hidden_layers
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers_input = [n_observations] + hidden_layers[:-1]
        layers_output = hidden_layers

        self.layers = nn.ModuleList()
        for input, output in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(input, output))

        self.mean = nn.Linear(layers_output[-1], n_actions)

        self.log_std = nn.Linear(layers_output[-1], n_actions)

        # weights initialization
        # for i in range(len(self.layers)):
        #     self.layers[i].weight.data.uniform_(-init_w, init_w)
        #     self.layers[i].bias.data.uniform_(-init_w, init_w)

        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)

        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = state_batch
        for i in range(len(self.layers)):
            output = self.layers[i](input)
            output = F.relu(output)
            input = output

        mean = self.mean(output)
        log_std = self.log_std(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def copy(self):

        copy = self.__class__(
            self.n_observations,
            self.n_actions,
            self.hidden_layers,
            self.init_w,
            self.log_std_min,
            self.log_std_max,
        )
        copy.load_state_dict(self.state_dict())
        return copy
