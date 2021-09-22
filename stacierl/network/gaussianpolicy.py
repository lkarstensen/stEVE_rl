import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .network import Network
from ..environment import ActionSpace


class GaussianPolicy(Network):
    def __init__(
        self,
        hidden_layers: List[int],
        action_space: ActionSpace,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)
        self.action_space = action_space
        self.hidden_layers = hidden_layers
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        n_actions = 1
        for dim in action_space.shape:
            n_actions *= dim

        layers_input = hidden_layers[:-1]
        layers_output = hidden_layers[1:]

        self.layers: List[nn.Linear] = nn.ModuleList()
        for input, output in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(input, output))

        self.mean = nn.Linear(hidden_layers[-1], n_actions)

        self.log_std = nn.Linear(hidden_layers[-1], n_actions)

        # weights initialization
        # for i in range(len(self.layers)):
        #     self.layers[i].weight.data.uniform_(-init_w, init_w)
        #     self.layers[i].bias.data.uniform_(-init_w, init_w)

        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)

        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    @property
    def input_is_set(self) -> bool:
        return len(self.layers) == len(self.hidden_layers)

    @property
    def n_inputs(self) -> Tuple[int, int]:
        return self.layers[0].in_features

    @property
    def n_outputs(self) -> int:
        return self.layers[-1].out_features

    def set_input(self, n_observations):
        n_output = self.hidden_layers[0]
        self.layers.insert(0, nn.Linear(n_observations, n_output))

    def forward(
        self, state_batch: PackedSequence, *args, **kwargs
    ) -> Tuple[PackedSequence, PackedSequence]:
        input, seq_length = pad_packed_sequence(state_batch, batch_first=True)
        for layer in self.layers:
            output = layer(input)
            output = F.relu(output)
            input = output

        mean = self.mean(output)
        log_std = self.log_std(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        mean = pack_padded_sequence(mean, seq_length, batch_first=True, enforce_sorted=False)
        log_std = pack_padded_sequence(log_std, seq_length, batch_first=True, enforce_sorted=False)
        return mean, log_std

    def copy(self):

        copy = self.__class__(
            self.hidden_layers,
            self.action_space,
            self.init_w,
            self.log_std_min,
            self.log_std_max,
        )
        return copy

    def reset(self) -> None:
        ...
