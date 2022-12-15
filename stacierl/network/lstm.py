from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import PackedSequence
from .network import Network


class LSTM(Network):
    def __init__(self, n_inputs: int, n_layer: int, n_nodes: int):
        super().__init__()
        self._n_inputs = n_inputs
        self.n_layer = n_layer
        self.n_nodes = n_nodes
        self.lstm = None
        self.hidden_state = None

        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=self.n_nodes,
            num_layers=self.n_layer,
            batch_first=True,
            bias=True,
        )
        # weight init
        # for name, param in self.named_parameters():
        #    if "bias" in name:
        #        nn.init.constant_(param, 0.0)
        # elif "weight" in name:
        #     w_xi, w_xf, w_xc, w_xo = param.chunk(4, 0)
        #     for weights in [w_xi, w_xf, w_xo]:
        #         nn.init.xavier_uniform_(weights, gain=nn.init.calculate_gain("sigmoid"))
        #     nn.init.xavier_uniform_(w_xc, gain=nn.init.calculate_gain("tanh"))

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self.lstm.hidden_size

    def forward(
        self, input_batch: torch.Tensor, use_hidden_state, *args, **kwargs
    ) -> torch.Tensor:
        if use_hidden_state:
            output, self.hidden_state = self.lstm.forward(
                input_batch, self.hidden_state
            )
        else:
            output, _ = self.lstm.forward(input_batch)
        return output

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self):
        self.hidden_state = None
