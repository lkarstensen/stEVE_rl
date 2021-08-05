import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .qnetwork import QNetwork


class QNetworkLSTM(QNetwork):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_layers: List[int],
        n_lstm_nodes: int,
        n_lstm_layer: int,
        init_w=3e-3,
    ):
        super().__init__(n_observations, n_actions, hidden_layers, init_w)

        self.n_lstm_nodes = n_lstm_nodes
        self.n_lstm_layer = n_lstm_layer

        self.lstm = nn.LSTM(
            input_size=n_observations,
            hidden_size=n_lstm_nodes,
            num_layers=n_lstm_layer,
            batch_first=True,
        )

        layers_input = [n_lstm_nodes + n_actions] + hidden_layers[:-1]
        layers_output = hidden_layers

        self.layers = nn.ModuleList()
        for input, output in zip(layers_input, layers_output):
            self.layers.append(nn.Linear(input, output))

        self.layers.append(nn.Linear(layers_output[-1], 1))

        # init weights and bias
        # for i in range(len(self.layers)):
        self.layers[-1].weight.data.uniform_(-init_w, init_w)
        self.layers[-1].bias.data.uniform_(-init_w, init_w)

        hidden_shape = (n_lstm_layer, 1, n_lstm_nodes)
        self._initial_hidden_state = (torch.zeros(hidden_shape), torch.zeros(hidden_shape))

    def forward(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        hidden_state_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        lstm_output, hidden_out = self.lstm(state_batch, hidden_state_batch)

        input = torch.cat([lstm_output, action_batch], dim=-1)
        for i in range(len(self.layers) - 1):
            output = self.layers[i](input)
            output = F.relu(output)
            input = output

        # output without relu
        q_value_batch = self.layers[-1](output)

        return q_value_batch, hidden_out

    def copy(self):

        copy = self.__class__(
            self.n_observations,
            self.n_actions,
            self.hidden_layers,
            self.n_lstm_nodes,
            self.n_lstm_layer,
            self.init_w,
        )
        copy.load_state_dict(self.state_dict())
        return copy
