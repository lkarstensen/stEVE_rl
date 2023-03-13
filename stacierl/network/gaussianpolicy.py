from copy import deepcopy
from typing import Optional, Tuple
import logging
from torch import nn
import torch

from .network import Network


class GaussianPolicy(Network):
    def __init__(
        self,
        base: Network,
        n_actions: int,
        input_embedder: Optional[Network] = None,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)

        self.base = base
        self.n_actions = n_actions
        self.input_embedder = input_embedder
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean = nn.Linear(base.n_outputs, n_actions)
        self.log_std = nn.Linear(base.n_outputs, n_actions)
        base.add_module("gaussian_mean", self.mean)
        base.add_module("gaussian_log_std", self.log_std)

        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)

    @property
    def n_inputs(self) -> int:
        if self.input_embedder is None:
            return self.base.n_inputs
        return self.input_embedder.n_inputs

    @property
    def n_outputs(self) -> Tuple[int, int]:
        return self.n_actions, self.n_actions

    @property
    def device(self) -> torch.device:
        return self.log_std.weight.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_embedder is not None:
            base_obs = self.input_embedder.forward(obs_batch)
        else:
            base_obs = obs_batch

        base_out = self.base.forward(base_obs)

        mean = self.mean(base_out)
        log_std = self.log_std(base_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_embedder is not None:
            base_obs = self.input_embedder.forward_play(obs_batch)
        else:
            base_obs = obs_batch

        base_out = self.base.forward_play(base_obs)

        mean = self.mean(base_out)
        log_std = self.log_std(base_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def copy(self):

        copy = deepcopy(self)
        return copy

    def reset(self) -> None:
        ...
