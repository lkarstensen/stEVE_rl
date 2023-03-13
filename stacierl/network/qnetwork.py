from copy import deepcopy
from typing import Optional, Tuple
from torch import nn
import torch

from .network import Network


class QNetwork(Network):
    def __init__(
        self,
        base: Network,
        n_actions: int,
        input_embedder: Optional[Network] = None,
        init_w=3e-3,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.base = base
        self.input_embedder = input_embedder
        self.init_w = init_w

        if input_embedder is not None:
            emb_out = input_embedder.n_outputs
            base_in = base.n_inputs
            if base_in == emb_out:
                base.n_inputs = base_in + n_actions
            elif base_in == emb_out + n_actions:
                ...
            else:
                raise ValueError(
                    f"{base.n_inputs=} does neither match {input_embedder.n_outputs=}, nor {input_embedder.n_outputs=}  + {n_actions=} "
                )

        self.base_to_q = nn.Linear(base.n_outputs, 1)
        base.add_module("base_to_q", self.base_to_q)

        self.base_to_q.weight.data.uniform_(-init_w, init_w)
        self.base_to_q.bias.data.uniform_(-init_w, init_w)

    @property
    def n_inputs(self) -> Tuple[int, int]:
        if self.input_embedder is None:
            return self.base.n_inputs - self.n_actions, self.n_actions
        return self.input_embedder.n_inputs, self.n_actions

    @property
    def n_outputs(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        return self.base_to_q.weight.device

    # pylint: disable=arguments-differ
    def forward(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:

        if self.input_embedder is not None:
            embedder_out = self.input_embedder.forward(obs_batch=obs_batch)
            base_obs = torch.dstack([embedder_out, action_batch])
        else:
            base_obs = torch.dstack([obs_batch, action_batch])

        base_out = self.base.forward(obs_batch=base_obs)
        q_value_batch = self.base_to_q.forward(base_out)
        return q_value_batch

    def forward_play(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.input_embedder is not None:
                embedder_out = self.input_embedder.forward_play(obs_batch=obs_batch)
                base_obs = torch.dstack([embedder_out, action_batch])
            else:
                base_obs = torch.dstack([obs_batch, action_batch])

            base_out = self.base.forward_play(obs_batch=base_obs)
            q_value_batch = self.base_to_q.forward(base_out)
        return q_value_batch

    def copy(self):

        return deepcopy(self)

    def reset(self) -> None:
        ...
