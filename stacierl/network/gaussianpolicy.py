import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Optional, Tuple

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

        self._initial_hidden_state = None

    def forward(
        self, state_batch: torch.Tensor, hidden_state_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input = state_batch
        for i in range(len(self.layers)):
            output = self.layers[i](input)
            output = F.relu(output)
            input = output

        mean = self.mean(output)
        log_std = self.log_std(output)
        self.logger.debug(f"mean: {mean}")
        self.logger.debug(f"log_std: {log_std}")
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, None

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
