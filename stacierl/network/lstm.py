import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import PackedSequence
from .network import Network


class LSTM(Network):
    def __init__(self, n_layer: int, n_nodes: int):
        super().__init__()

        self.n_layer = n_layer
        self.n_nodes = n_nodes
        self.lstm = None
        self.hidden_state = None

    @property
    def n_inputs(self) -> int:
        return self.lstm.input_size

    @property
    def n_outputs(self) -> int:
        return self.lstm.hidden_size

    @property
    def input_is_set(self) -> bool:
        return self.lstm is not None

    def set_input(self, n_input):
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=self.n_nodes, num_layers=self.n_layer)

    def forward(
        self,
        input_batch: PackedSequence,
    ) -> PackedSequence:

        output, self.hidden_state = self.lstm.forward(input_batch, self.hidden_state)
        return output

    def copy(self):

        copy = self.__class__(self.hidden_layers)
        return copy

    def reset(self):
        self.hidden_state = None
