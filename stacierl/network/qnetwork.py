import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .network import Network


class QNetwork(Network):
    def __init__(self, n_observations: int, n_actions: int, hidden_layers: List[int], init_w=3e-3):
        super().__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers
        self.init_w = init_w

        layers_input = [n_observations + n_actions] + hidden_layers[:-1]
        layers_output = hidden_layers

        self.layers = nn.ModuleList()
        for input, output in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(input, output))

        self.layers.append(nn.Linear(layers_output[-1], 1))

        # init weights and bias
        # for i in range(len(self.layers)):
        self.layers[-1].weight.data.uniform_(-init_w, init_w)
        self.layers[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        input = torch.cat([state_batch, action_batch], dim=1)
        for i in range(len(self.layers) - 1):
            output = self.layers[i](input)
            output = F.relu(output)
            input = output

        # output without relu
        q_value_batch = self.layers[-1](output)

        return q_value_batch

    def copy(self):

        copy = self.__class__(self.n_observations, self.n_actions, self.hidden_layers, self.init_w)
        copy.load_state_dict(self.state_dict())
        return copy
