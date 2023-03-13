from copy import deepcopy
import torch
from torch import nn
from .network import Network


class LSTM(Network):
    def __init__(self, n_inputs: int, n_layer: int, n_nodes: int):
        super().__init__()
        self.n_layer = n_layer
        self.n_nodes = n_nodes
        self.lstm = None
        self.hidden_state = None

        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=n_nodes,
            num_layers=n_layer,
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
        return self.lstm.input_size

    @n_inputs.setter
    def n_inputs(self, n_inputs: int) -> None:
        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=self.lstm.hidden_size,
            num_layers=self.lstm.num_layers,
            batch_first=True,
            bias=True,
        )

    @property
    def n_outputs(self) -> int:
        return self.lstm.hidden_size

    @property
    def device(self) -> torch.device:
        return self.lstm.all_weights[0][0].device

    def forward(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        output, _ = self.lstm.forward(obs_batch)
        return output

    def forward_play(self, flat_obs: torch.Tensor, *args, **kwds) -> torch.Tensor:
        with torch.no_grad():
            output, self.hidden_state = self.lstm.forward(flat_obs, self.hidden_state)
        return output

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self):
        self.hidden_state = None
