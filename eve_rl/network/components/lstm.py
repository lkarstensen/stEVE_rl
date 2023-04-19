from typing import List, Optional, Union
import torch
from torch import nn
from ..network import Network


class LSTM(Network):
    def __init__(
        self,
        n_layer: int,
        n_nodes: int,
        n_inputs: Optional[int] = None,
        output_layer_size: Optional[Union[int, List[int]]] = None,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_nodes = n_nodes
        self.init_w = init_w

        self._lstm: nn.LSTM = None
        self._output_layers: List[nn.Linear] = None

        if n_inputs is not None:
            self.n_inputs = n_inputs
        if output_layer_size is not None:
            self.output_layer_size = output_layer_size

        self._hidden_state = None

    @property
    def n_inputs(self) -> int:
        if self._lstm is None:
            return None
        return self._lstm.input_size

    @n_inputs.setter
    def n_inputs(self, n_inputs: int) -> None:
        if self._lstm is None:
            self._lstm = nn.LSTM(
                input_size=n_inputs,
                hidden_size=self.n_nodes,
                num_layers=self.n_layer,
                batch_first=True,
                bias=True,
            )
        elif n_inputs != self.n_inputs:
            raise ValueError(f"{self.n_inputs=} already set different than {n_inputs=}")

    @property
    def n_outputs(self) -> int:
        if self._output_layers is None:
            return self._lstm.hidden_size
        out = [layer.out_features for layer in self._output_layers]
        out = out[0] if len(out) == 1 else out
        return out

    @property
    def output_layer_size(self) -> int:
        if self._output_layers is None:
            return None
        return self.n_outputs

    @output_layer_size.setter
    def output_layer_size(self, output_layer_size: int) -> None:
        if self._output_layers is None:
            if isinstance(output_layer_size, int):
                output_layer_size = [output_layer_size]
            self._output_layers = nn.ModuleList()
            for n_outputs in output_layer_size:
                layer = nn.Linear(self._lstm.hidden_size, n_outputs)
                layer.weight.data.uniform_(-self.init_w, self.init_w)
                layer.bias.data.uniform_(-self.init_w, self.init_w)
                self._output_layers.append(layer)

        elif self.output_layer_size != output_layer_size:
            raise ValueError(
                f"{self.output_layer_size=} already set different than {output_layer_size=}"
            )

    @property
    def device(self) -> torch.device:  # pylint: disable=no-member
        return self.lstm.all_weights[0][0].device

    def forward(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        output, _ = self._lstm.forward(obs_batch)
        if self._output_layers is not None:
            state = [layer(state) for layer in self._output_layers]
            state = state[0] if len(state) == 1 else state
        return output

    def forward_play(self, obs_batch: torch.Tensor, *args, **kwds) -> torch.Tensor:
        with torch.no_grad():
            output, self._hidden_state = self._lstm.forward(
                obs_batch, self._hidden_state
            )
            if self._output_layers is not None:
                state = [layer(state) for layer in self._output_layers]
                state = state[0] if len(state) == 1 else state
        return output

    def reset(self):
        self._hidden_state = None
