from typing import Any, Dict, Generator, List, Optional, OrderedDict, Tuple
import numpy as np
from torch.distributions.normal import Normal
from .model import Model, ModelNetworks, ModelStateDicts
from .. import network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SACModelStateDicts(ModelStateDicts):
    q_net_1: Dict[str, torch.Tensor]
    q_net_2: Dict[str, torch.Tensor]
    target_q_net_1: Dict[str, torch.Tensor]
    target_q_net_2: Dict[str, torch.Tensor]
    policy_net: Dict[str, torch.Tensor]

    def __iter__(self):
        return iter(
            [self.q_net_1, self.q_net_2, self.target_q_net_1, self.target_q_net_2, self.policy_net]
        )

    def copy(self):
        return SACModelStateDicts(
            deepcopy(self.q_net_1),
            deepcopy(self.q_net_2),
            deepcopy(self.target_q_net_1),
            deepcopy(self.target_q_net_2),
            deepcopy(self.policy_net),
        )


@dataclass
class SACModelNetworks(ModelNetworks):
    q_net_1: network.QNetwork
    q_net_2: network.QNetwork
    target_q_net_1: network.QNetwork
    target_q_net_2: network.QNetwork
    policy_net: network.GaussianPolicy
    device: torch.device = torch.device("cpu")

    def soft_tau_update(self, sac_model_state_dicts: SACModelStateDicts, tau: float):
        for own_state_dict, other_state_dict in zip(self.state_dicts, sac_model_state_dicts):

            for own_param, other_param in zip(own_state_dict.values(), other_state_dict.values()):
                own_param.data.copy_(tau * other_param + (1 - tau) * own_param)

    def load_state_dicts(self, sac_model_state_dicts: SACModelStateDicts):
        self.q_net_1.load_state_dict(sac_model_state_dicts.q_net_1)
        self.q_net_2.load_state_dict(sac_model_state_dicts.q_net_2)
        self.target_q_net_1.load_state_dict(sac_model_state_dicts.target_q_net_1)
        self.target_q_net_2.load_state_dict(sac_model_state_dicts.target_q_net_2)
        self.policy_net.load_state_dict(sac_model_state_dicts.policy_net)

    @property
    def state_dicts(self) -> SACModelStateDicts:
        state_dicts = SACModelStateDicts(
            self.q_net_1.state_dict(),
            self.q_net_2.state_dict(),
            self.target_q_net_1.state_dict(),
            self.target_q_net_2.state_dict(),
            self.policy_net.state_dict(),
        )
        return state_dicts

    def __iter__(self):
        return iter(
            [self.q_net_1, self.q_net_2, self.target_q_net_1, self.target_q_net_2, self.policy_net]
        )


class SAC(Model):
    def __init__(
        self,
        q_net_1: network.QNetwork,
        q_net_2: network.QNetwork,
        target_q_net_2: network.QNetwork,
        target_q_net_1: network.QNetwork,
        policy_net: network.GaussianPolicy,
        learning_rate: float = 0.0007525,
    ) -> None:
        self._nets = SACModelNetworks(q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_net)

        self.learning_rate = learning_rate
        self._init_optimizer(learning_rate)
        self._init_alpha(learning_rate)

    @property
    def q_net_1(self) -> network.QNetwork:
        return self._nets.q_net_1

    @property
    def q_net_2(self) -> network.QNetwork:
        return self._nets.q_net_2

    @property
    def target_q_net_1(self) -> network.QNetwork:
        return self._nets.target_q_net_1

    @property
    def target_q_net_2(self) -> network.QNetwork:
        return self._nets.target_q_net_2

    @property
    def policy_net(self) -> network.GaussianPolicy:
        return self._nets.policy_net

    def _init_optimizer(self, learning_rate):
        self.q1_optimizer = optim.Adam(self._nets.q_net_1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self._nets.q_net_2.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self._nets.policy_net.parameters(), lr=learning_rate / 2)

    def _init_alpha(self, learning_rate):
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_rate)

    def get_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        with torch.no_grad():
            flat_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            mean, log_std, hidden_state_out = self._nets.policy_net(flat_state, hidden_state)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).numpy()
            return action, hidden_state_out

    @property
    def initial_hidden_state(self) -> Optional[torch.Tensor]:
        return None

    def to(self, device: torch.device):
        self.device = device
        self._nets.to(device)
        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def copy(self):
        copy = self.__class__(
            self._nets.q_net_1.copy(),
            self._nets.target_q_net_1.copy(),
            self._nets.q_net_2.copy(),
            self._nets.target_q_net_2.copy(),
            self._nets.policy_net.copy(),
            self.learning_rate,
        )

        return copy

    def copy_shared_memory(self):
        self._nets.q_net_1.share_memory()
        self._nets.target_q_net_1.share_memory()
        self._nets.q_net_2.share_memory()
        self._nets.target_q_net_2.share_memory()
        self._nets.policy_net.share_memory()
        copy = self.__class__(
            self._nets.q_net_1,
            self._nets.target_q_net_1,
            self._nets.q_net_2,
            self._nets.target_q_net_2,
            self._nets.policy_net,
            self.learning_rate,
        )

        return copy

    @property
    def nets(self) -> SACModelNetworks:
        return self._nets
